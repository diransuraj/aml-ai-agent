// Map labels
MATCH (e:Entity) SET e:Bank;
// Map relationships
MATCH (e1:Entity)<-[:ORIGINATOR]-(f:Filing)-[:BENEFITS]->(e2:Entity)
MERGE (e1)-[r:SENT_TO]->(e2)
SET r.amount = toFloat(f.amount), r.date = f.date;
// Map countries
MATCH (b:Bank)-[]-(c:Country) SET b.country = c.code;