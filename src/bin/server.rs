//! HTTP API server for ternary model inference.
//! Provides an OpenAI-compatible /v1/completions endpoint.
//!
//! Usage: cargo run --release --bin server -- --model models/my_model_1_58bit --port 8080

use std::env;
use std::io::Read;
use std::path::Path;
use std::process;
use std::sync::{Arc, Mutex};

use cpu_ai_framework::inference::format::PackedModel;
use cpu_ai_framework::inference::transformer::TernaryTransformer;
use cpu_ai_framework::inference::sampler::SamplingStrategy;
use cpu_ai_framework::inference::generate::{generate, GenerateConfig};
use cpu_ai_framework::inference::tokenizer::TokenizerWrapper;

use serde::{Deserialize, Serialize};
use tiny_http::{Server, Response, Header, Method};

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

struct AppState {
    model: TernaryTransformer,
    tokenizer: Option<TokenizerWrapper>,
    model_name: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path = String::new();
    let mut port: u16 = 8080;
    let mut host = String::from("127.0.0.1");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--port" => { i += 1; port = args[i].parse().unwrap_or(8080); }
            "--host" => { i += 1; host = args[i].clone(); }
            "--help" | "-h" => {
                println!("Usage: server --model <path> [--port <n>] [--host <addr>]");
                process::exit(0);
            }
            _ => { eprintln!("Unknown arg: {}", args[i]); process::exit(1); }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Error: --model is required.");
        process::exit(1);
    }

    println!("Loading model from: {}", model_path);
    let packed = PackedModel::load(&model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        process::exit(1);
    });

    let model_name = format!("ternary-{}-{}B",
        packed.manifest.model_type,
        packed.manifest.num_layers);

    println!("Building transformer...");
    let model = TernaryTransformer::from_packed(&packed);

    let model_dir = Path::new(&model_path);
    let tokenizer = TokenizerWrapper::from_file(model_dir.join("tokenizer.json")).ok();
    if tokenizer.is_some() {
        println!("Tokenizer loaded.");
    } else {
        eprintln!("Warning: No tokenizer found, using byte-level fallback.");
    }

    let state = Arc::new(Mutex::new(AppState { model, tokenizer, model_name }));

    let addr = format!("{}:{}", host, port);
    let server = Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to start server: {}", e);
        process::exit(1);
    });

    println!("Server listening on http://{}", addr);
    println!("Endpoints:");
    println!("  POST /v1/completions");
    println!("  GET  /health");

    for mut request in server.incoming_requests() {
        let path = request.url().to_string();
        let method = request.method().clone();

        match (method, path.as_str()) {
            (Method::Get, "/health") => {
                let resp = Response::from_string("{\"status\":\"ok\"}")
                    .with_header(json_header());
                let _ = request.respond(resp);
            }
            (Method::Post, "/v1/completions") => {
                let mut body = String::new();
                if request.as_reader().read_to_string(&mut body).is_err() {
                    let _ = request.respond(error_response(400, "Bad request body"));
                    continue;
                }

                let req: CompletionRequest = match serde_json::from_str(&body) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = request.respond(error_response(400, &format!("Invalid JSON: {}", e)));
                        continue;
                    }
                };

                let result = handle_completion(&state, req);
                match result {
                    Ok(resp_json) => {
                        let resp = Response::from_string(resp_json)
                            .with_header(json_header());
                        let _ = request.respond(resp);
                    }
                    Err(e) => {
                        let _ = request.respond(error_response(500, &e));
                    }
                }
            }
            _ => {
                let _ = request.respond(error_response(404, "Not found"));
            }
        }
    }
}

fn handle_completion(
    state: &Arc<Mutex<AppState>>,
    req: CompletionRequest,
) -> Result<String, String> {
    let mut state = state.lock().map_err(|e| format!("Lock error: {}", e))?;

    let prompt = req.prompt.unwrap_or_default();
    let max_tokens = req.max_tokens.unwrap_or(128);
    let temperature = req.temperature.unwrap_or(0.7);

    // Tokenize
    let (prompt_tokens, eos_id) = if let Some(ref tok) = state.tokenizer {
        let ids = tok.encode(&prompt, true).map_err(|e| format!("Tokenize error: {}", e))?;
        (ids, tok.eos_token_id)
    } else {
        (prompt.bytes().map(|b| b as u32).collect(), 2u32)
    };

    let strategy = if temperature < 0.01 {
        SamplingStrategy::Greedy
    } else if let Some(p) = req.top_p {
        SamplingStrategy::TopP { p, temperature }
    } else {
        SamplingStrategy::TopK { k: req.top_k.unwrap_or(40), temperature }
    };

    let config = GenerateConfig {
        max_tokens,
        strategy,
        eos_token_id: eos_id,
        doc_boundary_tokens: Vec::new(),
    };

    let result = generate(&mut state.model, &prompt_tokens, &config);

    // Decode
    let text = if let Some(ref tok) = state.tokenizer {
        tok.decode(&result.token_ids).unwrap_or_default()
    } else {
        let bytes: Vec<u8> = result.token_ids.iter()
            .filter_map(|&t| if t < 256 { Some(t as u8) } else { None })
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    };

    let finish_reason = if result.token_ids.last() == Some(&eos_id) {
        "stop"
    } else {
        "length"
    };

    let response = CompletionResponse {
        id: format!("cmpl-{}", rand::random::<u32>()),
        object: "text_completion".to_string(),
        model: state.model_name.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: result.token_ids.len(),
            total_tokens: prompt_tokens.len() + result.token_ids.len(),
        },
    };

    serde_json::to_string(&response).map_err(|e| format!("Serialize error: {}", e))
}

fn json_header() -> Header {
    Header::from_bytes("Content-Type", "application/json").unwrap()
}

fn error_response(code: u16, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = format!("{{\"error\":{{\"message\":\"{}\",\"code\":{}}}}}", msg, code);
    Response::from_string(body)
        .with_status_code(code)
        .with_header(json_header())
}
