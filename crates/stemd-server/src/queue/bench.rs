use stemd_core::{Audio, resample};

fn stem(secs: f64) -> Audio {
    let n = (secs * 44100.0) as usize;
    Audio::new(
        (0..2)
            .map(|c| {
                (0..n)
                    .map(|i| ((i as f32 * 0.01) + c as f32).sin() * 0.3)
                    .collect()
            })
            .collect(),
        44_100,
    )
}

/// Is converting three stems at once faster than one at a time? Not obviously:
/// the conversion can be memory-bound, and the 44.1 to 48 kHz path threads
/// internally already.
#[test]
#[ignore]
fn is_converting_three_stems_at_once_worth_it() {
    let stems: Vec<Audio> = (0..3).map(|_| stem(120.0)).collect();
    for target in [48_000u32, 96_000] {
        for round in 0..3 {
            let serial = std::time::Instant::now();
            for s in &stems {
                std::hint::black_box(resample::to_rate(s, target).unwrap());
            }
            let serial = serial.elapsed();

            let parallel = std::time::Instant::now();
            std::thread::scope(|scope| {
                let running: Vec<_> = stems
                    .iter()
                    .map(|s| scope.spawn(move || resample::to_rate(s, target).unwrap()))
                    .collect();
                for t in running {
                    std::hint::black_box(t.join().unwrap());
                }
            });
            let parallel = parallel.elapsed();
            println!(
                "  {target} Hz round {round}: serial {serial:>8.2?}  parallel {parallel:>8.2?}  \
                 ({:.2}x)",
                serial.as_secs_f64() / parallel.as_secs_f64()
            );
        }
    }
}
