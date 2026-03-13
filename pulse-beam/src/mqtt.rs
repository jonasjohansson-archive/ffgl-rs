use rumqttc::{Client, Event, MqttOptions, Packet, QoS};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub struct MqttHandle {
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MqttHandle {
    pub fn new(host: &str, port: u16, topic: &str, trigger: Arc<AtomicBool>) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let topic = topic.to_string();

        let mut opts = MqttOptions::new("pulsebeam", host, port);
        opts.set_keep_alive(Duration::from_secs(30));

        let (client, mut connection) = Client::new(opts, 10);
        let _ = client.subscribe(&topic, QoS::AtLeastOnce);

        let thread = thread::spawn(move || {
            for notification in connection.iter() {
                if shutdown_clone.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(Event::Incoming(Packet::Publish(_))) = notification {
                    trigger.store(true, Ordering::Relaxed);
                }
            }
        });

        MqttHandle {
            shutdown,
            thread: Some(thread),
        }
    }
}

impl Drop for MqttHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
