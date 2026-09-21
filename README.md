# Variational Autoencoder (VAE) for MNIST

> A VAE in PyTorch — the autoencoder made probabilistic, so it can generate new handwritten digits instead of only reconstructing existing ones.

A plain autoencoder maps each input to a single point in latent space. A VAE
maps it to a **distribution**, then samples from it. That one change is what
turns a compression tool into a generative model.

## Run it

Open [`Vartional_Auto_encoder.ipynb`](Vartional_Auto_encoder.ipynb).

```bash
pip install torch torchvision matplotlib numpy tqdm
```

MNIST downloads automatically.

## Configuration

| Parameter | Value |
|---|---|
| `batch_size` | 100 |
| `latent_dim` | 200 |
| `epochs` | 30 |
| Optimizer | Adam |

Structured as three modules — `Encoder`, `Decoder`, and a `Model` that wires
them together. Generated grids are written out with
`torchvision.utils.save_image` and `make_grid`.

## How it works

**Encoder** maps an image to the parameters of a Gaussian — a mean `μ` and a
log-variance `log σ²` — rather than to a single latent code.

**Sampling** draws `z` from that distribution. Done naively this blocks
gradients, so the VAE uses the **reparameterisation trick**:

```
z = μ + σ · ε,    ε ~ N(0, I)
```

The randomness is pushed into `ε`, which doesn't depend on the network
parameters, so `z` stays differentiable with respect to `μ` and `σ` and
backpropagation works normally. This trick is the reason VAEs are trainable at
all.

**Decoder** reconstructs an image from `z`.

## The loss

Two terms in tension:

```
L = reconstruction_loss + KL(q(z|x) ‖ N(0, I))
```

- **Reconstruction** pushes the output to match the input — the autoencoder's
  usual job.
- **KL divergence** pushes each encoded distribution toward a standard normal.

Without the KL term the model would shrink every variance to zero and degenerate
into an ordinary autoencoder: precise reconstructions, but a latent space full
of holes where sampling produces noise.

The KL term is what makes the latent space **continuous and complete**. Because
every input maps to a spread rather than a point, and all those spreads are
pulled toward the same prior, the regions between encoded digits decode into
plausible digits too. That's what makes generation work: sample `z ~ N(0, I)`,
run the decoder, get a new digit that was never in the training set.

## The trade-off

The two terms pull against each other, and where you land shows in the output.
Weight reconstruction too heavily and the latent space fragments. Weight KL too
heavily and everything collapses toward the prior — the classic VAE blur, where
samples look like an average digit rather than a specific one.

## Related

The deterministic baseline — linear and convolutional autoencoders on the same
dataset — is in
[Auto-encoder-](https://github.com/Fezzaioussama/Auto-encoder-).
