# Rust Safe Subset Development Guidelines

## 1. Core Mindset: Use "Ownership" Instead of "Manual Memory Management"

- **Use only `String` and `Vec`**: They are Rust's most essential RAII containers, automatically allocating and freeing memory.
- **Prefer `&T` (immutable borrowing)**: For everyday parameter passing and data reading, pass only references to avoid unnecessary deep copies (`.clone()`).
- **Minimize `&mut T` (mutable borrowing)**: Only use when truly needing to modify data. Maintaining immutability is the shortcut to writing safe code.
- **Avoid `unsafe`**: Never use `unsafe` blocks or raw pointers. Leave all memory safety concerns to the compiler.

## 2. Types & Interfaces: Keep It Simple

- **Use only `struct` and `enum`**: For organizing data, don't create complex generic constraints (Trait Bounds).
- **Use only basic Traits**: Such as `Debug`, `Clone`, `Copy`. Don't design complex Trait inheritance hierarchies. Avoid generic programming (Monomorphization) except for operator implementations.
- **Avoid complex macros**: Use only declarative macros like `#[derive(...)]` for automatic code generation. Never use `macro_rules!` or procedural macros.

## 3. Control Flow & Error Handling: Be Explicit

- **Use only `match` and `if let`**: For handling `Option` and `Result`, intuitive and safe.
- **Use only `?` operator**: This is Rust's most elegant error handling mechanism. Propagate errors directly upward, don't write complex error conversion logic.
- **Avoid advanced functional patterns**: Use only basic `.map()`, `.filter()`, `.iter()`. For slightly complex logic, write `for` loops straightforwardly. Never write nested iterator chains or closures.

## 4. Concurrency Model: Stability First

- **Use only `std::thread`**: Plain operating system-level threads, combined with `Arc<Mutex<T>>` for data sharing.
- **Avoid `async/await` ecosystem**: Rust's async model is extremely complex. If there's no high-concurrency performance bottleneck, never use `tokio` or `async`. Synchronous code is always the easiest to debug.