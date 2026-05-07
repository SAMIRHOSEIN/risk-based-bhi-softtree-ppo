
SELECT b0."8 - Structure Number" AS structure_number,
	   b0."Year" AS year0,
	   b1."Year" AS year1,
	   b0."58 - Deck Condition Rating" AS deck0,
	   b1."58 - Deck Condition Rating" AS deck1
FROM multnomah_bridges as b0
-- user LEFT JOIN if we are insterested in identifying end and gap years
INNER JOIN multnomah_bridges as b1
ON b0."8 - Structure Number" = b1."8 - Structure Number"
	AND b0."Year" = b1."Year"-1;