// 连续批处理器

use std::collections::{HashMap, VecDeque};

/// 请求状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    Running,
    Waiting,
    Preempted,
    Finished,
}

/// 请求上下文
#[derive(Debug)]
pub struct RequestContext {
    pub request_id: u64,
    pub state: RequestState,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
}

/// 连续批处理器
pub struct ContinuousBatcher {
    requests: HashMap<u64, RequestContext>,
    running: Vec<u64>,
    waiting: VecDeque<u64>,
    max_batch_size: usize,
}

impl ContinuousBatcher {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            requests: HashMap::new(),
            running: Vec::new(),
            waiting: VecDeque::new(),
            max_batch_size,
        }
    }

    pub fn add_request(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_tokens: usize,
    ) {
        let ctx = RequestContext {
            request_id: id,
            state: RequestState::Waiting,
            prompt_tokens: prompt,
            generated_tokens: Vec::new(),
            max_tokens,
        };
        self.requests.insert(id, ctx);
        self.waiting.push_back(id);
    }

    pub fn schedule(&mut self) {
        while self.running.len() < self.max_batch_size {
            if let Some(id) = self.waiting.pop_front() {
                if let Some(ctx) = self.requests.get_mut(&id) {
                    ctx.state = RequestState::Running;
                    self.running.push(id);
                }
            } else {
                break;
            }
        }
    }
}
