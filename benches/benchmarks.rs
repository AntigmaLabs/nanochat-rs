use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_template::utils;

fn benchmark_process_data(c: &mut Criterion) {
    c.bench_function("process_data", |b| {
        b.iter(|| utils::process_data(black_box("hello world")))
    });
}

fn benchmark_validate_input(c: &mut Criterion) {
    c.bench_function("validate_input", |b| {
        b.iter(|| utils::validate_input(black_box("test input")))
    });
}

criterion_group!(benches, benchmark_process_data, benchmark_validate_input);
criterion_main!(benches);
