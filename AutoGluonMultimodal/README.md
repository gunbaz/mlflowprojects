# Autogluon Multimodal (Tabular + Text)

## Goals
1) autogluon ile en az iki farklı veri türüyle örnek: tabular + text model yapma  
2) bunu MLflow'da gösterme  
3) kendini runtime'da update etme (parametre veya konfigürasyon ile)

## Yapı Önerisi
- `src/` veya `notebooks/` altında çalışmaları yap
- MLflow deney adı ve tag'leri burada belgeleyerek ilerle
- Gerekirse `config.yaml` ile parametreleri yönet

## Notlar
- Deney sonuçları, metrikler ve run URL'lerini bu dosyaya ekle.
