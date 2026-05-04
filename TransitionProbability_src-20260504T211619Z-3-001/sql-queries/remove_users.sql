-- Drop users if they already exist
DROP USER IF EXISTS seaclark, tianxlan, reilly8, stephe, nazario, mipurkey, jsalum2, altra2, ellav, infrarisk;

-- Automatically strip all other privileges granted to db_reader in this database
DROP OWNED BY db_reader;

-- Now drop the role
DROP ROLE IF EXISTS db_reader;