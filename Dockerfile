FROM python:3.10-bullseye

WORKDIR /app

# ========================================
# Install system dependencies
# ========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    wget \
    curl \
    procps \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Verify Java
RUN java -version

# ========================================
# Download Spark JARs (Delta Lake + XGBoost)
# ========================================
RUN mkdir -p /app/jars && \
    echo "📦 Downloading Delta Lake JARs..." && \
    wget -q -P /app/jars \
    https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.1.0/delta-spark_2.12-3.1.0.jar \
    https://repo1.maven.org/maven2/io/delta/delta-storage/3.1.0/delta-storage-3.1.0.jar && \
    echo "📦 Downloading XGBoost JARs..." && \
    wget -q -P /app/jars \
    https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/2.0.3/xgboost4j-spark_2.12-2.0.3.jar \
    https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/2.0.3/xgboost4j_2.12-2.0.3.jar && \
    echo "✅ JARs downloaded successfully"

# Set Spark classpath
ENV SPARK_CLASSPATH=/app/jars/*

# ========================================
# Create data directories
# ========================================
RUN mkdir -p /app/data/raw \
    /app/data/bronze \
    /app/data/silver \
    /app/data/gold \
    /app/configs \
    /app/models/pipeline \
    /app/models/production

# ========================================
# Install Python dependencies
# ========================================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ========================================
# Copy application code
# ========================================
COPY . .

# Default command
CMD ["bash"]