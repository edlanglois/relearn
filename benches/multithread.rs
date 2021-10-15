use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::{mpsc, Arc, Barrier, Mutex, RwLock};
use std::thread;

fn barrier_mutex(num_threads: usize, num_iterations: u64, vector_size: usize) -> u64 {
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let mut buffers = Vec::new();
    let sum = Arc::new(RwLock::new(0u64));

    for i in 0..num_threads {
        let thread_barrier = Arc::clone(&barrier);
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(vector_size)));
        let thread_buffer = Arc::clone(&buffer);
        let thread_sum = Arc::clone(&sum);
        buffers.push(buffer);

        let i_ = ((i + 1) % 32) as u64;
        thread::spawn(move || {
            let mut key;
            for j in 0..num_iterations {
                key = *thread_sum.read().unwrap() % 32768;
                let mut buffer_guard = thread_buffer.lock().unwrap();
                // Fill the buffer
                let j_ = (j + 1) % 32;
                for k in 0..vector_size {
                    buffer_guard.push(i_ * j_ * ((k % 32) as u64) + key);
                }
                drop(buffer_guard);
                // Wait for all threads to finish filling their buffers
                thread_barrier.wait();
                // Wait for the main thread to finish processing the buffers
                thread_barrier.wait();
            }
        });
    }

    for _ in 0..num_iterations {
        // Wait for all threads to finish filling their buffers
        barrier.wait();
        // Update sum
        let mut sumg = sum.write().unwrap();
        for buffer in &buffers {
            for elem in buffer.lock().unwrap().drain(..) {
                *sumg = sumg.wrapping_add(elem);
            }
        }
        // Indicate that the processing is done
        barrier.wait();
    }
    let sum = *sum.read().unwrap();
    sum
}

fn channels(num_threads: usize, num_iterations: u64, vector_size: usize) -> u64 {
    // let sync_pair = Arc::new(Mutex::new(0usize) /*sync count*/, Condvar::new());

    let mut send_empty_buffer_channels = Vec::new();
    let (send_full_buffer, recv_full_buffers) = mpsc::sync_channel(0);

    let sum = Arc::new(RwLock::new(0u64));

    for i in 0..num_threads {
        // let thread_sync_pair = Arc::clone(&sync_pair);

        let (send_empty_buffer, thread_recv_empty_buffer) = mpsc::sync_channel(0);
        send_empty_buffer_channels.push(send_empty_buffer);
        let thread_send_full_buffer = send_full_buffer.clone();
        let thread_sum = Arc::clone(&sum);

        let i_ = ((i + 1) % 32) as u64;
        thread::spawn(move || {
            let mut key;

            for j in 0..num_iterations {
                // Update key from sum
                key = *thread_sum.read().unwrap() % 32768;

                // Get a buffer from the main thread
                let mut buffer: Vec<u64> = thread_recv_empty_buffer.recv().unwrap();
                // let mut sync_count = thread_sync_pair.0.lock().unwrap();
                // *sync_count += 1;
                // if sync_count == num_threads {
                //     thread_sync_pair.notify_all();
                // }

                // Fill the buffer
                let j_ = (j + 1) % 32;
                for k in 0..vector_size {
                    buffer.push(i_ * j_ * ((k % 32) as u64) + key);
                }

                // Send the buffer (along with thread_id for returning it)
                thread_send_full_buffer.send((buffer, i)).unwrap();
            }
        });
    }

    let mut buffers = vec![Some(Vec::with_capacity(vector_size)); num_threads];
    for _ in 0..num_iterations {
        // Send a buffer to each thread
        for (channel, buffer) in send_empty_buffer_channels.iter().zip(&mut buffers) {
            channel.send(buffer.take().unwrap()).unwrap();
        }

        // Update sum
        let mut sumg = sum.write().unwrap();
        for _ in 0..num_threads {
            let (mut buffer, i) = recv_full_buffers.recv().unwrap();

            for elem in buffer.drain(..) {
                *sumg = sumg.wrapping_add(elem);
            }

            buffers[i] = Some(buffer);
        }
    }

    let sum = *sum.read().unwrap();
    sum
}

/// Test different methods of sharing buffers between a manager and worker threads.
fn multithread_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("multithread_buffers");

    let num_threads = 4;
    let num_iterations = 1000;
    let vector_size = 100;

    group.bench_function("barrier_mutex", |b| {
        b.iter(|| barrier_mutex(num_threads, num_iterations, vector_size))
    });
    group.bench_function("channels", |b| {
        b.iter(|| channels(num_threads, num_iterations, vector_size))
    });
}

criterion_group!(benches, multithread_buffers);
criterion_main!(benches);
