FROM python:3.9.2-alpine3.13

# Environment variables
ENV PACKAGES=/usr/local/lib/python3.9/site-packages
ENV PYTHONDONTWRITEBYTECODE=1

# Set build directory
WORKDIR /tmp

COPY requirements.txt .

RUN set -e ;\
    apk upgrade --update-cache -a ;\
    apk add --no-cache libstdc++ libffi-dev ;\
    apk add --no-cache --virtual .build gcc g++ musl-dev python3-dev cargo openssl-dev git;\
    pip install --no-cache-dir -r requirements.txt

# clean 
RUN apk del .build ;\
    rm -rf /tmp/* /root/.cache
    
# Set working directory
WORKDIR /docs

# Expose MkDocs development server port
EXPOSE 8000

ENV PYTHONPATH=$PYTHONPATH:/docs/python
# Start development server by default
ENTRYPOINT ["mkdocs"]
CMD ["serve", "--dev-addr=0.0.0.0:8000"]