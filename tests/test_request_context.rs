// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test request context.

#[cfg(test)]
mod tests {
    use infer_train::backend::MemoryRegionId;
    use infer_train::inference::{
        Priority, RequestContext, RequestScratchPool, RequestState,
    };

    #[test]
    fn test_request_context_creation() {
        let region = MemoryRegionId(0);
        let prompt = vec![1, 2, 3, 4, 5];
        let request = RequestContext::new(
            1,
            prompt.clone(),
            10,
            0.7,
            Priority::Normal,
            region,
        );

        assert_eq!(request.request_id, 1);
        assert_eq!(request.state, RequestState::Waiting);
        assert_eq!(request.prompt_tokens, prompt);
        assert_eq!(request.prompt_pos, 0);
        assert!(request.generated_tokens.is_empty());
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.temperature, 0.7);
        assert_eq!(request.priority, Priority::Normal);
        assert!(!request.is_prefill_complete());
    }

    #[test]
    fn test_request_advance() {
        let region = MemoryRegionId(0);
        let prompt = vec![1, 2, 3];
        let mut request =
            RequestContext::new(1, prompt, 10, 0.7, Priority::Normal, region);

        assert_eq!(request.current_token(), Some(1));
        request.advance_prompt();
        assert_eq!(request.current_token(), Some(2));
        request.advance_prompt();
        assert_eq!(request.current_token(), Some(3));
        request.advance_prompt();
        assert!(request.is_prefill_complete());
    }

    #[test]
    fn test_request_generation() {
        let region = MemoryRegionId(0);
        let prompt = vec![1];
        let mut request =
            RequestContext::new(1, prompt, 3, 0.7, Priority::Normal, region);

        // Complete prefill
        request.advance_prompt();
        assert!(request.is_prefill_complete());

        // Generate tokens
        request.add_generated_token(10);
        request.add_generated_token(20);
        assert_eq!(request.generated_tokens.len(), 2);
        assert!(!request.is_complete());

        request.add_generated_token(30);
        assert!(request.is_complete());
    }

    #[test]
    fn test_scratch_pool() {
        let region = MemoryRegionId(0);
        let mut pool = RequestScratchPool::new(4096, 16384, region);

        // Preallocate 4 buffers
        pool.preallocate(4);
        assert_eq!(pool.available(), 4);

        // Allocate one
        let scratch1 = pool.allocate().unwrap();
        assert_eq!(scratch1.hidden.len(), 4096);
        assert_eq!(scratch1.ffn_gate.len(), 16384);
        assert_eq!(pool.available(), 3);

        // Release it
        pool.release(scratch1);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::RealTime > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }
}
