# PgAdmin and TimescaleDB
## How to access to pgadmin
Goto http://1.233.218.4:9913/

## How to view data in pgadmin
At the menubar on the left, go through TimescaleDB -> Databases -> `[DatabaseName(=PerSite_DB)]` -> Schemas -> `[SchemaName(=public)]` -> Tables -> `[TableName]`, right-click the table, select View/Edit Data -> All rows

## DB structure
At Database=PerSite_DB, Schema=public, there are two tables.
1. `idu_t`\
It contains the idu information of the site.
    |id|name|metadata|
    |--|----|--------|
    |integer|varchar(50)|varchar(255)|

2. `data_t`\
It contains the real data of each idu, timestamp.\

    |id|idu_id|roomtemp|settemp|oper|timestamp|
    |--|------|--------|-------|----|---------|
    |integer|integer|double|double|bool|timestamp without timezone|

## Information of the currently stored data
Processed version of YongDongIllHighSchool collected from 7 idus during 2022.08.01-2022.09.30. It is interpolated to have sampling rate of 60 secs/sample.