-- Create the reader role if it doesn't exist
CREATE ROLE db_reader;

-- Secure the schema: Only owners/admins can create new objects (tables, views)
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- Grant select on all current tables
GRANT USAGE ON SCHEMA public TO db_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO db_reader;

-- Ensure future tables are also readable
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO db_reader;

-- CREATE USERS
CREATE USER seaclark WITH PASSWORD 'pkh+Igm49u' IN ROLE db_reader;
CREATE USER tianxlan WITH PASSWORD 'rr4x@q9Kvm' IN ROLE db_reader;
CREATE USER reilly8 WITH PASSWORD 'zwb9$Kmg7h' IN ROLE db_reader;
CREATE USER stephe WITH PASSWORD 'tv9faJ#4at' IN ROLE db_reader;
CREATE USER nazario WITH PASSWORD 'hP7nwv-k2z' IN ROLE db_reader;
CREATE USER mipurkey WITH PASSWORD 'bsbkB8k*9e' IN ROLE db_reader;
CREATE USER jsalum2 WITH PASSWORD 'kqee5$N9wv' IN ROLE db_reader;
CREATE USER altra2 WITH PASSWORD 'u7#Fz8jnem' IN ROLE db_reader;
CREATE USER ellav WITH PASSWORD 'afazS8+q8s' IN ROLE db_reader;
CREATE USER infrarisk WITH PASSWORD 'te66_45M' IN ROLE db_reader;