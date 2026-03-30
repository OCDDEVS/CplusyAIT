//! Chat prompt templates for instruction-tuned models.
//! Formats user/system/assistant messages into the expected token format.

/// A single message in a conversation.
pub struct ChatMessage {
    pub role: String,  // "system", "user", "assistant"
    pub content: String,
}

/// Supported chat template formats.
pub enum ChatFormat {
    /// Llama 3 / 3.1 instruct format
    Llama3,
    /// Llama 2 chat format
    Llama2,
    /// ChatML format (used by many models)
    ChatML,
    /// Raw — no template, just concatenate content
    Raw,
}

impl ChatFormat {
    /// Auto-detect format from model_type string.
    pub fn from_model_type(model_type: &str) -> Self {
        match model_type.to_lowercase().as_str() {
            s if s.contains("llama") => {
                // Llama 3+ uses the new format
                ChatFormat::Llama3
            }
            s if s.contains("chatml") || s.contains("qwen") || s.contains("phi") => {
                ChatFormat::ChatML
            }
            _ => ChatFormat::Llama3, // Default to Llama3 format
        }
    }

    /// Format a list of messages into a single prompt string.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatFormat::Llama3 => format_llama3(messages),
            ChatFormat::Llama2 => format_llama2(messages),
            ChatFormat::ChatML => format_chatml(messages),
            ChatFormat::Raw => messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n"),
        }
    }
}

fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    // Signal the model to generate an assistant response
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn format_llama2(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut system_msg = None;

    for msg in messages {
        match msg.role.as_str() {
            "system" => system_msg = Some(msg.content.clone()),
            "user" => {
                out.push_str("<s>[INST] ");
                if let Some(ref sys) = system_msg {
                    out.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
                    system_msg = None;
                }
                out.push_str(&format!("{} [/INST]", msg.content));
            }
            "assistant" => {
                out.push_str(&format!(" {} </s>", msg.content));
            }
            _ => {}
        }
    }
    out
}

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
