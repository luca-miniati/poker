import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
# Setup duckdb connection
DB_PATH = '../data/db/master.db'
con = duckdb.connect(DB_PATH)
# phhs data is split into 3 tables 
con.sql('show tables;').show()
# `hands` contains a row of all the scalar fields from each parsed hand
con.sql('select * from hands limit 5;').show()
# `players` contains a row for each player in each hand, containing information specific to each player
con.sql("select * from players where hand_id = '26505473230';").show()
# Note that this table is not a list of all players.
# Rather, it stores player-specific information for each hand.
# To illustrate this, we can query from this table all the hands that the player `XGMcz0rzod8mj+/PpWmQeQ` participated in:
con.sql("select distinct hand_id from players where name = 'XGMcz0rzod8mj+/PpWmQeQ';")
# `actions` contains a row for each action in each hand
con.sql("select * from actions where hand_id = '26941575934';").show()
# Okay, enough with the basics. Let's answer some questions.
# Q1: Who are the most active players on the site?

query = '''
select
    p.name,
    count(*) as cnt
from players p join hands h on p.hand_id = h.hand_id
group by p.name
order by cnt desc
limit 5;
'''

con.sql(query).show()
# Q2: What does the distribution of hand lengths look like?

query = '''
select
    num_actions,
    count(*) as cnt
from hands
group by num_actions
'''

num_actions_df = con.sql(query).df()
num_actions_df.head()
plt.figure(figsize=(20,6))
sns.barplot(x='num_actions', y='cnt', data=num_actions_df, color='skyblue')
# Q3: What does the distribution of VPIP's look like?

query = '''
with flop_index as (
    select hand_id, min(action_index) as flop_action_index
    from actions
    where action_type = 'db' and length(cards) = 6
    group by hand_id
),

preflop_actions as (
    select
        a.hand_id,
        p.name as player_name,
        coalesce(a.amount, 0) as amount,
        coalesce(p.blind_or_straddle, 0) as blind_or_straddle
    from actions a
    join players p
      on a.hand_id = p.hand_id and a.actor = 'p' || cast(p.player_idx as varchar)
    left join flop_index f
      on a.hand_id = f.hand_id
    where a.action_index < coalesce(f.flop_action_index, -1)
),

vpip as (
    select
        hand_id,
        player_name,
        case when sum(case when amount > blind_or_straddle then 1 else 0 end) > 0
             then 1 else 0 end as vpip
    from preflop_actions
    group by hand_id, player_name
)

select
    player_name,
    avg(coalesce(vpip, 0)) as vpip,
    count(distinct hand_id) as sample_size
from vpip
group by player_name
having sample_size >= 200
order by random();
'''

df_vpip = con.sql(query).df()
df_vpip.head()
plt.figure(figsize=(10,6))
sns.kdeplot(df_vpip['vpip'], fill=True, color='skyblue')
plt.title('VPIP (KDE)')
plt.xlabel('VPIP')
plt.ylabel('Density')
plt.show()
con.close()