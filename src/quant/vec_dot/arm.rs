pub mod fp;
pub mod iq;
pub mod iq_v2;
pub mod iq_v3;
pub mod iq_v4;
pub mod iq_v5;
pub mod iq_v6;
pub mod iq_v7;
pub mod iq_v8;
pub mod k_quant;
pub mod q_0_1;
pub mod tq;

pub use fp::{
    fp16_to_fp32, vec_dot_bf16_neon_fallback, vec_dot_fp16_neon,
    vec_dot_fp32_neon,
};

#[cfg(target_feature = "bf16")]
pub use fp::vec_dot_bf16_neon;

pub use iq::{
    vec_dot_iq1_m_q8_k_neon, vec_dot_iq1_s_q8_k_neon, vec_dot_iq2_s_q8_k_neon,
    vec_dot_iq2_xs_q8_k_neon, vec_dot_iq2_xxs_q8_k_neon,
    vec_dot_iq3_s_q8_k_neon, vec_dot_iq3_xxs_q8_k_neon,
    vec_dot_iq4_nl_q8_0_neon, vec_dot_iq4_xs_q8_k_neon,
};

pub use iq_v2::vec_dot_iq3_xxs_q8_k_neon_v2;
pub use iq_v3::vec_dot_iq3_xxs_q8_k_neon_v3;
pub use iq_v4::vec_dot_iq3_xxs_q8_k_neon_v4;
pub use iq_v5::vec_dot_iq3_xxs_q8_k_neon_v5;
pub use iq_v6::vec_dot_iq3_xxs_q8_k_neon_v6;
pub use iq_v7::vec_dot_iq3_xxs_q8_k_neon_v7;
pub use iq_v8::vec_dot_iq3_xxs_q8_k_neon_v8;

pub use k_quant::{
    vec_dot_q2_k_q8_k_neon, vec_dot_q3_k_q8_k_neon, vec_dot_q4_k_q8_k_neon,
    vec_dot_q5_k_q8_k_neon, vec_dot_q6_k_q8_k_neon,
};

pub use q_0_1::{
    vec_dot_q4_0_q8_0_neon, vec_dot_q4_1_q8_1_neon, vec_dot_q5_0_q8_0_neon,
    vec_dot_q5_1_q8_1_neon, vec_dot_q8_0_q8_0_neon,
};

pub use tq::{vec_dot_tq1_0_q8_k_neon, vec_dot_tq2_0_q8_k_neon};
