FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları (git + curl for grype)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Grype Vulnerability Scanner (for SBOM scanning)
RUN curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin || true

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Modeli eğit
CMD ["python", "train.py"]

