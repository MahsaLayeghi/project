# Train Your First Neural Network in PyTorch

Fit a small MLP to the moons dataset and drive training loss below 0.35.

## Layout

```
data.py    given — moons dataset factory
model.py   YOU implement MoonsNet
train.py   YOU implement the training loop
tests/     given — do not modify
```

## Quickstart

```bash
bash run.sh  # trains and saves results.json
bash test.sh # pytest
```

## Reflection

> Replace this section with your own reflection. Explain in 3–5 sentences what each of these five lines is doing in your training loop:
>
> ```python
> optimiser.zero_grad()
> preds = model(X)
> loss = criterion(preds, y)
> loss.backward()
> optimiser.step()
> ```
# project
# project
