"""
Tabular + Text veri seti oluşturma modülü
E-ticaret ürün incelemelerini simüle eder
"""
import pandas as pd
import numpy as np

def generate_multimodal_dataset(n_samples=500):
    """
    Hem tabular hem de text verisi içeren dataset oluşturur
    """
    np.random.seed(42)
    
    # Tabular features
    prices = np.random.uniform(10, 500, n_samples)
    ratings = np.random.uniform(1, 5, n_samples)
    num_reviews = np.random.randint(0, 1000, n_samples)
    discount = np.random.uniform(0, 50, n_samples)
    
    # Text features - ürün açıklamaları ve yorumlar
    positive_words = ['excellent', 'amazing', 'great', 'perfect', 'love', 'best', 'fantastic', 'wonderful']
    negative_words = ['terrible', 'bad', 'worst', 'disappointed', 'poor', 'awful', 'hate', 'useless']
    neutral_words = ['product', 'item', 'purchase', 'order', 'delivery', 'package', 'received', 'quality']
    
    reviews = []
    categories = []
    labels = []
    
    for i in range(n_samples):
        # Rating'e göre yorum oluştur
        if ratings[i] > 3.5:
            sentiment = 'positive'
            words = np.random.choice(positive_words, size=np.random.randint(3, 8))
            label = 1
        elif ratings[i] < 2.5:
            sentiment = 'negative'
            words = np.random.choice(negative_words, size=np.random.randint(3, 8))
            label = 0
        else:
            sentiment = 'neutral'
            words = np.random.choice(neutral_words, size=np.random.randint(3, 8))
            label = 1 if np.random.random() > 0.5 else 0
            
        review_text = ' '.join(words) + f" This product is {sentiment}."
        reviews.append(review_text)
        
        # Kategori
        category = np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'])
        categories.append(category)
        
        labels.append(label)
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'price': prices,
        'rating': ratings,
        'num_reviews': num_reviews,
        'discount': discount,
        'category': categories,
        'review_text': reviews,
        'label': labels  # 1: satın alınır, 0: satın alınmaz
    })
    
    return df

if __name__ == "__main__":
    # Küçük bir örnek oluştur (10 satır)
    df = generate_multimodal_dataset(n_samples=10)
    
    print("\n" + "="*120)
    print("📊 ÜRETİLEN VERİ ÖRNEKLERİ (E-TİCARET ÜRÜN İNCELEMELERİ)")
    print("="*120 + "\n")
    
    # Her satırı detaylı göster
    for idx, row in df.iterrows():
        print(f"🔹 ÖRNEK #{idx + 1}:")
        print(f"   💰 Fiyat:         {row['price']:.2f} TL")
        print(f"   ⭐ Puan:          {row['rating']:.2f}/5.0")
        print(f"   💬 Yorum Sayısı:  {row['num_reviews']} adet")
        print(f"   🏷️  İndirim:       %{row['discount']:.1f}")
        print(f"   📦 Kategori:      {row['category']}")
        print(f"   📝 Müşteri Yorumu: \"{row['review_text']}\"")
        print(f"   🎯 TAHMİN SONUCU: {'✅ ALIR (1)' if row['label'] == 1 else '❌ ALMAZ (0)'}")
        print()
    
    print("="*120)
    print("\n📈 VERİ İSTATİSTİKLERİ:")
    print(f"   • Toplam Örnek:     {len(df)}")
    print(f"   • Alır (1):         {(df['label'] == 1).sum()} örnek ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"   • Almaz (0):        {(df['label'] == 0).sum()} örnek ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"\n   • Ortalama Fiyat:   {df['price'].mean():.2f} TL")
    print(f"   • Ortalama Puan:    {df['rating'].mean():.2f}/5.0")
    print(f"   • Ortalama Yorum:   {df['num_reviews'].mean():.0f} adet")
    print(f"   • Ortalama İndirim: %{df['discount'].mean():.1f}")
    
    print("\n📋 VERİ ÇERÇEVESİ (DataFrame Görünümü):")
    print("-"*120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=True))
    
    print("\n💾 CSV DOSYASINA KAYDET:")
    df.to_csv('sample_data.csv', index=False)
    print("   ✓ Veri 'sample_data.csv' dosyasına kaydedildi!")
    
    print("\n" + "="*120)
    print("✨ VERİ YAPISI:")
    print("="*120)
    print(df.info())
    print("="*120 + "\n")
