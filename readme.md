# Docker
## How to run all the services
`docker compose up -d`
## How to run specific service
`docker compose up -d [service name]`

## 위 방식을 응용해 같은 machine에서 여러명의 개발자가 git으로 따로 작업하기
`docker-compose.yml`에서 `main_s` 서비스의 컨테이너 이름을 중복되지 않도록 `main_yourname_s`등으로 바꾼 후 
`docker compose up -d main_s`로 실행하면 컨테이너 conflict이 일어나지 않음. 

# PgAdmin and TimescaleDB
## How to access to pgadmin
Goto http://1.233.218.4:9913/

## How to view data in pgadmin
At the menubar on the left, go through TimescaleDB -> Databases -> `[DatabaseName(=PerSite_DB)]` -> Schemas -> `[SchemaName(=public)]` -> Tables -> `[TableName]`, right-click the table, select View/Edit Data -> All rows

## DB structure
Database=PerSite_DB, Schema=public에는 두 개의 테이블이 있음.
1. `idu_t`\
이 테이블에는 사이트의 IDU(실내기) 정보가 포함되어 있음.
    |id|name|metadata|
    |--|----|--------|
    |integer|varchar(50)|varchar(255)|

2. `data_t`\
이 테이블에는 각 IDU의 실제 데이터와 타임스탬프가 포함되어 있음.

    |id|idu_id|roomtemp|settemp|oper|timestamp|
    |--|------|--------|-------|----|---------|
    |integer|integer|double|double|bool|timestamp without timezone|

## Information of the currently stored data
2022년 8월 1일부터 2022년 9월 30일까지 7개의 IDU에서 수집된 영동일고등학교(YongDongIllHighSchool) 데이터의 3_Processed 버전. 샘플링 주기가 60초/샘플로 보간되었음.

Hi i am sungmin

# Demo scenario
## Metadata
```
{"site_name": "YongDongIllHighSchool",
"user_name": "홍길동",
"user_role": "customer", # customer, admin
"current_datetime": "2022-09-30 12:00:00",}
```
asdADAdA  AASddaFASDFSA