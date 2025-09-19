FROM registry.sberdevices.ru/registry/tools/python:3.11.4

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --index-url https://nexus.sberdevices.ru/repository/Pypi/simple/ --no-cache-dir -r requirements.txt

COPY entrypoint.py .

EXPOSE 8080

CMD ["python3", "entrypoint.py"]
