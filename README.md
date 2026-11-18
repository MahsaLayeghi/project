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
> the training loop consits of 5 steps:
> 1. Zero the gradient: it is important to zero the gradient because pyTorch adds the gradient by default. so we clear it 
> 2. Forward Pass: we will push the input data to the model to get a prediction
> 3. Calculate Loss: compare the prediction value with an actual value with a loss function. we want to find out how wrong the model is.
> 4. Backward Pass: we call backward on the loss to calculate the gradient of the loss. we will find out how much nudging each weight changed the loss.
> 5. Optimiser Step: then the optimiser nudger every weight in the direction that lowers the loss.
> ```python
> optimiser.zero_grad()
> preds = model(X)
> loss = criterion(preds, y)
> loss.backward()
> optimiser.step()
> ```
# project
# project
