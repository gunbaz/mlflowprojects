"""
Simple LLM Chatbot using Hugging Face GPT-2
Bu basit chatbot, Garak ve PyRIT güvenlik testleri için hedef model olarak kullanılacak.
"""

from transformers import pipeline, set_seed
import warnings
warnings.filterwarnings('ignore')

class SimpleLLM:
    def __init__(self, model_name="gpt2"):
        """
        Basit GPT-2 chatbot başlatıcı
        
        Args:
            model_name: Hugging Face model adı (default: gpt2)
        """
        print(f"[INFO] Loading model: {model_name}...")
        self.generator = pipeline(
            'text-generation',
            model=model_name,
            device=-1  # CPU kullan (GPU gerektirmez)
        )
        set_seed(42)  # Reproducibility için
        print("[OK] Model loaded successfully!")
    
    def generate(self, prompt, max_length=100, num_return_sequences=1):
        """
        Prompt'a göre metin üret
        
        Args:
            prompt: Giriş metni
            max_length: Maksimum çıktı uzunluğu
            num_return_sequences: Kaç farklı yanıt üretilecek
            
        Returns:
            Üretilen metin(ler)
        """
        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            if num_return_sequences == 1:
                return outputs[0]['generated_text']
            else:
                return [output['generated_text'] for output in outputs]
                
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return None
    
    def chat(self, user_input):
        """
        Basit chat interface
        
        Args:
            user_input: Kullanıcı girdisi
            
        Returns:
            Model yanıtı
        """
        # Prompt formatla
        prompt = f"User: {user_input}\nAssistant:"
        
        # Yanıt üret
        response = self.generate(prompt, max_length=150)
        
        # Sadece Assistant kısmını al
        if response and "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
            return assistant_response
        
        return response


def test_chatbot():
    """
    Chatbot'u test et
    """
    print("\n" + "="*60)
    print("Simple LLM Chatbot Test")
    print("="*60)
    
    # Chatbot başlat
    llm = SimpleLLM()
    
    # Test promptları
    test_prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a joke.",
    ]
    
    for prompt in test_prompts:
        print(f"\n[USER] {prompt}")
        response = llm.chat(prompt)
        print(f"[BOT] {response}")
    
    print("\n" + "="*60)
    print("[OK] Chatbot test completed!")


if __name__ == "__main__":
    test_chatbot()
