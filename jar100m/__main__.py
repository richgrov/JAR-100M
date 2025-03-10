import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from jar100m.dataset import Dataset
from jar100m.model import Model

CONTEXT_WINDOW_SIZE = 64
EPOCHS = 32
LOSS_REPORT_INTERVAL = 2000
BATCH_SIZE = 16

def get_batch(dataset, indices):
    return jnp.stack([dataset.get_item(i)[0] for i in indices]), \
           jnp.stack([dataset.get_item(i)[1] for i in indices])

def cross_entropy_loss(logits, classes):
    batches, context_size, probs = logits.shape
    logits = logits.reshape(batches * context_size, probs)
    classes = classes.reshape(batches * context_size)
    return optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(classes, probs)).mean()

@eqx.filter_jit
def compute_loss(model, inp, expected_outp):
    pred_logits = model(inp)
    return cross_entropy_loss(pred_logits, expected_outp)

@eqx.filter_value_and_grad
def loss_and_grad(model, inp, expected_outp):
    return compute_loss(model, inp, expected_outp)

def validate(model, dataset, indices):
    inp, expected_outp = get_batch(dataset, indices)
    return compute_loss(model, inp, expected_outp)

def count_parameters(model):
    params = eqx.filter(model, eqx.is_array)
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    return total_params

def main():
    with open("dataset2.txt", 'r') as file:
        shakespeare = file.read()

    dataset = Dataset(shakespeare, CONTEXT_WINDOW_SIZE)
    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)
    
    model = Model(len(dataset.vocab), CONTEXT_WINDOW_SIZE, model_key)
    model = eqx.filter_jit(eqx.filter_vmap(model))
    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    num_params = count_parameters(model)
    print(f"Model has {num_params} parameters")

    train_size = int(0.2 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_indices = jax.random.permutation(train_key, train_size)
    val_indices = jax.random.permutation(train_key, val_size)

    train_loss_history = []
    validate_loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        print(train_size)
        for i in range(0, train_size, BATCH_SIZE):
            batch_indices = train_indices[i:i+BATCH_SIZE]
            inp, expected_outp = get_batch(dataset, batch_indices)
            
            loss, grads = loss_and_grad(model, inp, expected_outp)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            
            total_loss += loss.item()
            
            if i % LOSS_REPORT_INTERVAL == 0 and i > 0:
                avg_loss = total_loss / (LOSS_REPORT_INTERVAL / BATCH_SIZE)
                val_loss = validate(model, dataset, val_indices[:BATCH_SIZE])
                print(f"Epoch {epoch}, step {i}: train loss {avg_loss}, validate loss {val_loss}")
                train_loss_history.append(avg_loss)
                validate_loss_history.append(val_loss.item())
                total_loss = 0

    sequence = dataset.encode("\n")[None, :]
    gen_key = jax.random.PRNGKey(42)

    for _ in range(1000):
        logits = model(sequence)[:, -1]
        gen_key, subkey = jax.random.split(gen_key)
        next_token = jax.random.categorical(subkey, logits)
        print(dataset.decode(next_token), end="", flush=True)
        sequence = jnp.concatenate([sequence, next_token[None, :]], axis=1)

    plt.plot(train_loss_history, label="Train loss")
    plt.plot(validate_loss_history, label="Validate loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
