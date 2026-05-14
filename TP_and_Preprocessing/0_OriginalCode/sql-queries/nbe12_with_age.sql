SELECT nbe.*, 2025 - nbi."BW01" as age
FROM "PDX24"."8-Elements" AS nbe
INNER JOIN "PDX24"."1-Primary" AS nbi
USING ("BID01")
WHERE "BE01"=12