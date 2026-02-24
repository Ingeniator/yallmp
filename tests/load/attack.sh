#!/bin/bash

# Пароль и организация (можно хранить в секретах)
export USER="your_user"
export PASS="your_password"
export ORG_ID="your_org_id"
export HOSTNAME="https://hostname"

# Запускаем Artillery
export FILE_PATH="filepath_to_postdata"

# uncomment if you want to see all request and response
# export DEBUG="http,http:request,http:response,plugin:expect"

npx artillery run ./artillery.yaml  --environment functional #--output report.json #-o result.json #--quiet 
exit $?
