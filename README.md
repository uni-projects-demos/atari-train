# Atari Train

Train, eval and export Atari policies to ONNX using ALE and Burn with DQN, IQN, PPO or Rainbow for:

* Boxing
* Breakout
* Ice Hockey
* Pong
* Space Invaders
* Tennis

# Install

```sh
chmod +x crates/ale-bridge/native/setup.sh
./crates/ale-bridge/native/setup.sh
```

# Build

```sh
cargo build --locked --release
```

# Run

```sh
cargo run --locked --release -- [OPTIONS]
```

### Train

```sh
cargo run --locked --release -- train --game <game> --alg <algorithm>
```

### Train with Config

```bash
cargo run --locked --release -- train --config configs/<game>.json
```

### Train with GPU

```bash
cargo run --locked --release --features gpu -- train [OPTIONS] --device gpu
```

# Evaluate

```bash
cargo run --locked --release -- eval --checkpoint runs/<game>/<checkpoint>.safetensors [OPTIONS]
```

# Export

```bash
cargo run --locked --release -- export --checkpoint runs/<game>/<checkpoint>.safetensors --output <file-name>.onnx
```

# Contribute

Before making an Issue-linked Pull Request, verify it passes CI checks:

```bash
cargo fmt --all
cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features
cargo check --locked --workspace --all-features
```
