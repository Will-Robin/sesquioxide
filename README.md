# sesquioxide

A text-frequency inverse document frequency (TF-IDF) based search engine for
nested directories of `.txt` and `.md` files.

This is something I put together to learn Rust.

Sesquioxide is a rust pun. Al<sub>2</sub>O<sub>3</sub> is a sesquioxide,
as is Fe<sub>2</sub>O<sub>3</sub>.

## Building/Installing/Testing

First, you'll need to have [Rust
installed](https://www.rust-lang.org/tools/install), and this repository cloned
to your computer.

Navigate to the directory containing this code. For a simple demo, run `cargo
run` (hint: try typing in 'sky' when prompted). Run `cargo build --release` to
compile a binary. The compiled program will be saved in `target/release`. It can
then be copied to wherever you like and run from the command line:
`./sesquioxide`. Alternatively, use `cargo install --path .` for a system wide
installation.

Run tests with `cargo test`. One may fail, depending on the system (the success
of the test relies upon the order in which files names are read).

A directory can be given as an argument e.g. `./sesquioxide <dir>` (the default
is the current directory). It will recursively search through all the
directories within that directory, so don't run it from your home directory, for
example, otherwise you are in for a long wait.
