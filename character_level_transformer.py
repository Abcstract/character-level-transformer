import math
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------
# 1. Basic settings
# -----------------------------

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

batch_size = 32
context_length = 128
training_steps = 5000
learning_rate = 3e-4
evaluation_steps = 10

model_width = 256
number_of_heads = 8
number_of_layers = 6
dropout_rate = 0.1

print("Using device:", device)


# -----------------------------
# 2. Read and encode the text
# -----------------------------

with open("wizard_of_oz.txt", "r", encoding="utf-8") as file:
    text = file.read()

characters = sorted(set(text))
vocabulary_size = len(characters)

character_to_number = {character: number for number, character in enumerate(characters)}
number_to_character = {number: character for number, character in enumerate(characters)}


def encode(text_to_encode):
    """Turn characters into integer IDs."""
    return [character_to_number[character] for character in text_to_encode]


def decode(numbers):
    """Turn integer IDs back into characters."""
    return "".join(number_to_character[number] for number in numbers)


all_data = torch.tensor(encode(text), dtype=torch.long)
split_point = int(0.9 * len(all_data))
training_data = all_data[:split_point]
validation_data = all_data[split_point:]

print(f"Characters: {len(text):,}")
print(f"Vocabulary size: {vocabulary_size}")


def get_batch(data):
    """Get random input sequences and the next-character targets."""
    starting_points = torch.randint(
        len(data) - context_length - 1,
        (batch_size,),
    )

    inputs = torch.stack([
        data[start:start + context_length]
        for start in starting_points
    ])

    targets = torch.stack([
        data[start + 1:start + context_length + 1]
        for start in starting_points
    ])

    return inputs.to(device), targets.to(device)


# -----------------------------
# 3. Transformer building blocks
# -----------------------------

class SelfAttention(nn.Module):
    """Several attention heads that can only look at earlier characters."""

    def __init__(self):
        super().__init__()

        if model_width % number_of_heads != 0:
            raise ValueError("model_width must be divisible by number_of_heads")

        self.head_size = model_width // number_of_heads
        self.number_of_heads = number_of_heads

        # One layer creates queries, keys, and values all at once.
        self.make_qkv = nn.Linear(model_width, 3 * model_width, bias=False)
        self.output = nn.Linear(model_width, model_width, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size_here, sequence_length, width = x.shape

        query, key, value = self.make_qkv(x).chunk(3, dim=-1)

        # Change from (batch, sequence, width) to
        # (batch, heads, sequence, head_size).
        query = query.view(batch_size_here, sequence_length, self.number_of_heads, self.head_size)
        key = key.view(batch_size_here, sequence_length, self.number_of_heads, self.head_size)
        value = value.view(batch_size_here, sequence_length, self.number_of_heads, self.head_size)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # is_causal=True prevents a position from looking into the future.
        # Dropout must be zero during evaluation, especially on Apple MPS.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=dropout_rate if self.training else 0.0,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size_here, sequence_length, width)
        attended = self.output(attended)

        return self.dropout(attended)


class TransformerBlock(nn.Module):
    """One attention layer followed by one feed-forward layer."""

    def __init__(self):
        super().__init__()

        self.first_norm = nn.LayerNorm(model_width)
        self.attention = SelfAttention()

        self.second_norm = nn.LayerNorm(model_width)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_width, 4 * model_width),
            nn.GELU(),
            nn.Linear(4 * model_width, model_width),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # Residual connections let information skip around each sub-layer.
        x = x + self.attention(self.first_norm(x))
        x = x + self.feed_forward(self.second_norm(x))
        return x


# -----------------------------
# 4. The language model
# -----------------------------

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabulary_size, model_width)
        self.position_embedding = nn.Embedding(context_length, model_width)

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(number_of_layers)]
        )

        self.final_norm = nn.LayerNorm(model_width)
        self.output_layer = nn.Linear(model_width, vocabulary_size, bias=False)

        # Use the same weights for reading and predicting characters.
        self.output_layer.weight = self.token_embedding.weight

        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None):
        _, sequence_length = input_ids.shape

        token_vectors = self.token_embedding(input_ids)
        positions = torch.arange(sequence_length, device=device)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocabulary_size),
                target_ids.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, starting_ids, number_of_new_characters, temperature=0.7, top_k=40):
        self.eval()
        output_ids = starting_ids

        for _ in range(number_of_new_characters):
            recent_ids = output_ids[:, -context_length:]
            logits, _ = self(recent_ids)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                best_logits, _ = torch.topk(
                    next_logits,
                    min(top_k, next_logits.size(-1)),
                )
                cutoff = best_logits[:, [-1]]
                next_logits[next_logits < cutoff] = float("-inf")

            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            output_ids = torch.cat((output_ids, next_id), dim=1)

        return output_ids


# -----------------------------
# 5. Train and evaluate
# -----------------------------

model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1,
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for name, data in [("train", training_data), ("validation", validation_data)]:
        losses = []

        for _ in range(evaluation_steps):
            inputs, targets = get_batch(data)
            _, loss = model(inputs, targets)
            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results


def learning_rate_at(step):
    """Warm up at the beginning, then slowly lower the learning rate."""
    warmup_steps = 500
    smallest_learning_rate = learning_rate / 10

    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, training_steps - warmup_steps)
    cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
    return smallest_learning_rate + cosine_value * (
        learning_rate - smallest_learning_rate
    )


for step in range(training_steps):
    current_learning_rate = learning_rate_at(step)
    for group in optimizer.param_groups:
        group["lr"] = current_learning_rate

    if step % 500 == 0:
        losses = estimate_loss()
        print(
            f"step {step:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"validation loss {losses['validation']:.4f}"
        )

    inputs, targets = get_batch(training_data)
    _, loss = model(inputs, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


# -----------------------------
# 6. Save and generate text
# -----------------------------

with open("language_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved to language_model.pkl")

prompt = "Once upon a time"
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated_ids = model.generate(prompt_ids, number_of_new_characters=300)
print("\n--- Generated text ---\n")
print(decode(generated_ids[0].tolist()))