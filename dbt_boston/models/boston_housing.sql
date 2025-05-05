SELECT 
    CAST("b" as NUMERIC) as b,
    CAST("rm" as NUMERIC) as rm,
    CAST("zn" as NUMERIC) as zn,
    CAST("age" as NUMERIC) as age,
    CAST("dis" as NUMERIC) as dis,
    CAST("nox" as NUMERIC) as nox,
    CAST("rad" as NUMERIC) as rad,
    CAST("tax" as NUMERIC) as tax,
    CAST("chas" as INTEGER) as chas,
    CAST("crim" as NUMERIC) as crim,
    CAST("medv" as NUMERIC) as medv,
    CAST("indus" as NUMERIC) as indus,
    CAST("lstat" as NUMERIC) as lstat,
    CAST("ptratio" as NUMERIC) as ptratio
FROM {{ source('boston_project_raw', 'boston_housing') }}



