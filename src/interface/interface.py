'''
JOTTING DOWN MY MIND
-----------------------

we have train_data: its a csv file which stores data of paths joining two hubs.
the format of saving data is like: 
data,source_number,destination_number,is_carting,is_ftl,day_of_week,start_hour,is_day,is_night,is_cutoff,osrm_time,osrm_distance,actual_distance,actual_time,real_actual_time_factor
training,0,1,1,0,3,3,0,1,0,44.0,54.2181,39.38604027413606,68.0,1.5454545454545454
we can get source/destination name simply using source/destination number, so even if there is no name in this data, we can get the names using the source and destination number

our graph model which is fully trained, takes the input as : "training,0,1,1,0,3,3,0,1,0,44.0,54.2181,39.38604027413606,68.0,1.5454545454545454" (same as the single row if train_data csv file)
of these things, the model will only take: source_number, destination_number, is_carting, is_ftl, start_hour, osrm_time into consideration, and it predicts the delay factor, once we multiply
that delay factor with osrm_time, we get our ETA.

how i want my interface.py to run:
- it should be streamlit designed, where someone can choose source name and destination name
i also have a csv file of all the source and destinations (they all are part of the same group, one hub can be source, and might also act as destination in some other travel)
hub_name,hub_number,bottleneck_score,delay_ratio,delay_hours,sla_breach,sla_breach_contribution,betweenness_centrality,closeness_centrality,degree_centrality
Anand_VUNagar_DC (Gujarat),0,0.054602,1.250607427976397,0.0338,0.0,0.001574582034652544,0.00041639974375400383,0.1004685003007058,0.0012812299807815502
this is what my hub-features.csv file's heading and first column looks like
so, if we have source name, we can get source number and vice versa easily!

- after user chooses these two things, they have option to choose the correct path from hub A to hub B. there are multiple ways to choose the path, so i want to apply diakstra to get
the best route. even if hub A and hub B are connected via direct path, there could be option to go indirectly, so i want all paths to be explored.
- if user defines the route type (ftl or carting) then only use that, but if he doesnt, then choose whatever has lower ETA (also, if i have to go from hub A to hub B via hub C, and hub A to hub B is better via ftl, and hub B to C is better via carting, then use both ftl and carting both one after one)
- there should be a sidebar, which defines "blacklisted hubs" and if one of the hubs in the path of hub A to hub B is blacklisted, then the model cannot choose that path, and suggest next better path

steps to follow:
1. first of all, i need to create a proper graph, and doing it in C++ seems better (doing DSA in c++ will definately help)
2. the c++ graph and python runner.py model need to talk with each other continously, instead of edge weight values in normal c++ graph, it will talk to runner.py to get the ETA
3. then ill apply diakstra into it, to get the best route via connection of surrounding hubs.
'''

