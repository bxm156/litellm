use litellm_cost::{PromptConvention, Usage, catalog};
use litellm_host_python::release_gil;
use pyo3::{prelude::*, pybacked::PyBackedBytes};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogRequest {
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    cache_write_5m_tokens: Option<u64>,
    cache_write_1h_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    service_tier: Option<String>,
    billed_at_ns: Option<i64>,
    threshold_is_inclusive: Option<bool>,
}

#[pyfunction]
pub fn calculate_catalog_cost(
    py: Python<'_>,
    pricing_json: PyBackedBytes,
    request_json: PyBackedBytes,
) -> Option<(f64, f64)> {
    release_gil(py, || {
        let fields: Map<String, Value> = serde_json::from_slice(&pricing_json).ok()?;
        let request: CatalogRequest = serde_json::from_slice(&request_json).ok()?;
        let cost = catalog::calculate(
            &fields,
            &catalog::EstimateRequest {
                usage: Usage {
                    prompt_tokens: request.prompt_tokens,
                    completion_tokens: request.completion_tokens,
                    cache_read_tokens: request.cache_read_tokens,
                    cache_write_tokens: request.cache_write_tokens,
                    cache_write_5m_tokens: request.cache_write_5m_tokens,
                    cache_write_1h_tokens: request.cache_write_1h_tokens,
                    prompt_convention: PromptConvention::IncludesCache,
                },
                service_tier: request.service_tier.as_deref(),
                reasoning_tokens: request.reasoning_tokens,
                billed_at_ns: request.billed_at_ns,
                threshold_is_inclusive: request.threshold_is_inclusive,
            },
        )?;
        Some((cost.input, cost.output))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytes;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::counts(json!({}), Some((120.0, 20.0)))]
    #[case::negative(json!({"prompt_tokens":-1}), None)]
    #[case::fractional(json!({"completion_tokens":1.5}), None)]
    #[case::inconsistent_cache(json!({"cache_write_1h_tokens":1}), None)]
    #[case::unknown_field(json!({"unknown":1}), None)]
    fn python_json_projects_shared_catalog_cost(
        #[case] overrides: Value,
        #[case] expected: Option<(f64, f64)>,
    ) {
        let fields = br#"{"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":2}"#;
        let mut request = json!({"prompt_tokens":100,"completion_tokens":10,"cache_read_tokens":0,"cache_write_tokens":20,"cache_write_5m_tokens":20,"cache_write_1h_tokens":0,"reasoning_tokens":0,"service_tier":null,"billed_at_ns":0,"threshold_is_inclusive":false}).as_object().unwrap().clone();
        request.extend(overrides.as_object().unwrap().clone());
        let serialized = serde_json::to_vec(&request).unwrap();
        Python::initialize();
        Python::attach(|py| {
            let result = crate::native_module(py)
                .getattr("calculate_catalog_cost")
                .unwrap()
                .call1((PyBytes::new(py, fields), PyBytes::new(py, &serialized)))
                .unwrap()
                .extract::<Option<(f64, f64)>>()
                .unwrap();
            assert_eq!(result, expected);
            let invalid = crate::native_module(py)
                .getattr("calculate_catalog_cost")
                .unwrap()
                .call1((PyBytes::new(py, b"invalid"), PyBytes::new(py, &serialized)))
                .unwrap();
            assert!(invalid.is_none());
        });
    }
}
