#!/bin/bash

# Включаем режим измерения времени выполнения
TIMEFORMAT='%lR'

# Настройки по умолчанию
MIN_SIZE=$((90*1024*1024))    # Минимальный размер файла (байты)
MIN_COUNT=20                  # Минимальное количество диалогов
MIN_LENGTH=$((200*1024))      # Минимальная длина одной строки

# Вспомогательные переменные
COUNT=0            # Реальное количество диалогов
TEXT_LENGTH=0      # Динамическая длина текста

# Проверка аргументов командной строки
while [[ "$#" -gt 0 ]]; do case $1 in
  --MIN_SIZE) MIN_SIZE="$2"; shift;;
  --min-count) MIN_COUNT="$2"; shift;;
  --max-length) MIN_LENGTH="$2"; shift;;
esac; shift; done

# Рассчитываем среднее потребление памяти на одну запись
AVG_PER_RECORD_MIN_SIZE=$((MIN_LENGTH * 3))

# Подсчет реального числа элементов массива
REQUIRED_COUNT=$(( (MIN_SIZE + AVG_PER_RECORD_MIN_SIZE - 1) / AVG_PER_RECORD_MIN_SIZE ))

# Проверка условий совместимости параметров
if [[ $REQUIRED_COUNT -lt $MIN_COUNT ]]; then
  echo "Ошибка: невозможно удовлетворить требования к размеру файла и количеству элементов одновременно." >&2
  exit 1
else
  COUNT=$REQUIRED_COUNT
fi

# Генерация случайного текста произвольной длины
generate_random_text_tr() {
  local len=$1
  LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c "$len"
}

generate_random_text() {
  local len=$1
  if ! command -v openssl &>/dev/null; then
    echo "Требуется наличие утилиты 'openssl'" >&2
    return 1
  fi
  openssl rand -hex $len | awk '{print toupper(substr($0, 1, length))}'
}

generate_json() {
# Формирование начального блока JSON
cat << EOF > dataset.json
{
  "dataset": [
EOF

# Создание содержимого массива
for ((i=0; i<$COUNT; i++)); do
  TEXT_LENGTH=$(((RANDOM % (2 * MIN_LENGTH - MIN_LENGTH + 1) + MIN_LENGTH) /2)) # Случайная длина строки от 1 MIN_LENGTH до 2 MIN_LENGTH
  cat << EOF >> dataset.json
    [
      {
        "role": "system",
        "content": "$(generate_random_text $TEXT_LENGTH)"
      },
      {
        "role": "user",
        "content": "$(generate_random_text $TEXT_LENGTH)"
      },
      {
        "role": "assistant",
        "content": "$(generate_random_text $TEXT_LENGTH)"
      }
    ],
EOF
done

# Удаляем завершающую запятую перед последним элементом
# Выбор правильной команды для получения размера файла
case "$(uname)" in
  Darwin*)  # macOS/BSD-like systems
    gsed -i '$ s/,//g' dataset.json
    ;;
  *)        # Other platforms like Linux
    sed -i '$ s/,//g' dataset.json
    ;;
esac

# Окончательная структура JSON
cat << EOF >> dataset.json
  ]
}
EOF
}

# Начало замера времени выполнения
start_time=$SECONDS

generate_json

# Получаем фактический размер файла
FILE_SIZE=$(wc -c < dataset.json)

# Итоговая информация
echo "Файл dataset.json успешно создан:"
echo "- Количество диалогов: $COUNT"
echo "- Средний размер диалога: примерно $(($FILE_SIZE/$COUNT))"
echo "- Фактический размер файла: $FILE_SIZE байт ($(($FILE_SIZE/(1048576))) Mб)"

# Завершаем замер времени выполнения
end_time=$SECONDS
total_time=$(( end_time - start_time ))
echo "- Время выполнения скрипта: ${total_time}s"
