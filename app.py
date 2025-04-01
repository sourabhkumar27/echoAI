from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, stripe, os

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model_name = "TheBloke/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>echoAI</h1><p>Your AI assistant is live.</p>
    <form action='/chat' method='post'><input name='prompt'><button>Ask</button></form>
    """

@app.post("/chat")
async def chat(request: Request):
    form = await request.form()
    prompt = form.get("prompt")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

@app.post("/create-checkout-session")
async def checkout():
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{"price": os.getenv("STRIPE_PRICE_ID"), "quantity": 1}],
        mode="subscription",
        success_url="https://echoai.com/success",
        cancel_url="https://echoai.com/cancel",
    )
    return {"url": session.url}
