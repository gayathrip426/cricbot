# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import html
import time
from fastapi.middleware.cors import CORSMiddleware
import os
import sqlite3

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch
from sklearn.preprocessing import LabelEncoder
import joblib
import openai
import spacy
import re

# FastAPI setup
app = FastAPI()

origins = ["http://localhost:5173"]  # Replace as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load intent classification model
intent_model_path = "./fine_tuned_models/fine_tuned_intent_model"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path)

# Load NER model
ner_model = spacy.load("./fine_tuned_models/cricket_ner_model/model-best")

# Load label encoder
le = joblib.load("label_encoder.pkl")

# Intent classifier
def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1).item()
    return predicted, probs

# NER
def named_entity_recog(text):
    results = ner_model(text)
    print(f"\nSentence: {text}")
    print(results)
    for r in results.ents:
        print(f"  {r.text} - {r.label_}")
    return results.ents

openai_client = openai.OpenAI(api_key="")

def generate_openai_response(prompt: str):
    try:
        # Use OpenAI's GPT-3 or GPT-4 (or other available engines)
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  # Use appropriate model (e.g., "gpt-4", "gpt-3.5-turbo")
                messages=[
                            {"role": "system", "content": "You are a helpful cricket assistant."},
                            {"role": "user", "content": prompt}
                         ],
            max_tokens=200,
            temperature=0.7,
            stop=["</s>"]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Request model
class Message(BaseModel):
    message: str

chat_history = []

def rule_based_ner(query):
    """Recognize the intent of the user query"""
    # This is a placeholder - replace with your actual intent recognition logic
    # For demonstration, we'll use a simple keyword matching approach
    
    categories = {}
    query_lower = query.lower()

    def parse_n(val):
        return int(val) if val.isdigit() else None
    
    for word in ["score", "runs", "centuries", "sixes", "fours","strike-rate","strike rate"]:
        if word in query_lower:
            categories['bat_bowl'] = {"name": "bat", "stat": word}
            break
    for word in ["wickets", "maidens", "overs", "economy"]:
        if word in query_lower:
            categories['bat_bowl'] = {"name": "bowl", "stat": word}
            break
    for word in ["catches","runouts","run-outs","stumpings"]:
        if word in query_lower:
            categories['bat_bowl'] = {"name": "field", "stat": word}
            break
    if any(word in query_lower for word in ["compare", "versus", "vs", "better", "comparison"]):
        categories['compare'] = True
    for word in ["highest","lowest","average","total","top","bottom"]:
        if word in query_lower:
            categories['high_low'] = word
            top_match = re.search(r'\b(?:top|highest)\s+(\d+)', query_lower)
            bottom_match = re.search(r'\b(?:bottom|lowest)\s+(\d+)', query_lower)
            if top_match:
                print(top_match.group(0))
                print(top_match.group(1))
                categories["top_n"] = parse_n(top_match.group(1))
            if bottom_match:
                categories["bottom_n"] = parse_n(bottom_match.group(1))
            break
    return categories

def construct_player_stat_query(player_name, stat, aggregation=None, tournament=None, year=None):
    # Map natural language stats to DB column names
    stat_map = {
        "runs": "runs_scored",
        "score": "runs_scored",
        "scorers": "runs_scored",
        "wickets": "wickets",
        "fours": "fours",
        "sixes": "sixes",
        "catches": "catches",
        "strike rate": "strike_rate",
        "strike-rate": "strike_rate",
        "strike_rate": "strike_rate",
        "economy": "economy",
        "centuries": "centuries"
    }
    if not player_name:
        return ""
    if not stat_map.get(stat, None):
        return ""
    # Resolve stat column
    stat_column = stat_map.get(stat.lower(), stat)

    # Determine aggregation function
    agg_map = {
        "total": f"SUM({stat_column})",
        "average": f"AVG({stat_column})",
        "highest": f"MAX({stat_column})",
        "lowest": f"MIN({stat_column})",
        None: stat_column
    }
    select_expr = agg_map.get(aggregation.lower() if aggregation else None)

    # Base query
    query = f"""
    SELECT {select_expr} AS result
    FROM player_match_stats pms
    JOIN players p ON pms.player_id = p.player_id
    JOIN matches m ON pms.match_id = m.match_id
    JOIN tournaments t ON m.tournament_id = t.tournament_id
    WHERE p.name LIKE '%{player_name}%'
    """

    # Add tournament filter
    if tournament:
        query += f" AND t.name LIKE '%{tournament}%'"

    # Add year filter
    if year:
        query += f" AND t.year = {year}"

    return query.strip()

def construct_tournament_stat_query(stat_type, tournament_name=None,year=None,top_n=None,bottom_n=None,include_player_names=True):
    # Base select
    select_clause = f"""
    SELECT 
        p.name AS player_name,
        t.name AS team_name,
        tr.name AS tournament_name,
        tr.year,
        SUM(ps.{stat_type}) AS total_{stat_type}
    """ if include_player_names else f"""
    SELECT 
        tr.name AS tournament_name,
        tr.year,
        SUM(ps.{stat_type}) AS total_{stat_type}
    """

    from_clause = """
    FROM player_match_stats ps
    JOIN players p ON ps.player_id = p.player_id
    JOIN teams t ON ps.team_id = t.team_id
    JOIN matches m ON ps.match_id = m.match_id
    JOIN tournaments tr ON m.tournament_id = tr.tournament_id
    """

    where_clauses = []
    if tournament_name:
        where_clauses.append(f"tr.name = '{tournament_name}'")
    if year:
        where_clauses.append(f"tr.year = {year}")

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    group_by_clause = """
    GROUP BY ps.player_id
    """

    order_by_clause = ""
    limit_clause = ""
    if top_n:
        order_by_clause = f"ORDER BY total_{stat_type} DESC"
        limit_clause = f"LIMIT {top_n}"
    elif bottom_n:
        order_by_clause = f"ORDER BY total_{stat_type} ASC"
        limit_clause = f"LIMIT {bottom_n}"

    # Final query
    query = f"""
    {select_clause}
    {from_clause}
    {where_clause}
    {group_by_clause}
    {order_by_clause}
    {limit_clause}
    """.strip()

    return query

def construct_db_query(entities: List[dict[str, any]], intent:str, prompt):
    query_params = []
    query = None
    player_name = None
    stat_entity = None 
    format_entity = None
    tournament = None
    year = None
    venue_entity = None
    rule_based_results = rule_based_ner(prompt)
    aggregation = rule_based_results.get('high_low', None)
    stat = rule_based_results.get('bat_bowl',{}).get('stat', None)
    top_n = rule_based_results.get('top_n',None)
    bottom_n = rule_based_results.get('bottom_n', None)
    for e in entities:
        #print(type(e["label_"]))
        if str(e.label_) == "PLAYER":
            player_name = e.text
        elif str(e.label_) == "STAT_TYPE":
            stat_entity = e.text
        elif str(e.label_) == "FORMAT":
            format_entity = e.text
        elif str(e.label_) == "TOURNAMENT":
            tournament = e.text
        elif str(e.label_) == "YEAR":
            year = e.text
        elif str(e.label_) == "VENUE":
            venue_entity = e.text
    if format_entity:
            tournament = format_entity
    if intent == "player_information" or  (player_name and stat):
        query = construct_player_stat_query(player_name, stat, aggregation=aggregation, tournament=tournament, year=year)
        print(query)
    if tournament and stat:
        query = construct_tournament_stat_query(stat, tournament,year,top_n=top_n,bottom_n=bottom_n,include_player_names=True)
        print(query)
    return query, query_params

def setup_database(db_path="./cricket_stats.db"):
    conn = sqlite3.connect(db_path) 
    cursor = conn.cursor()
    return cursor

def generate_response(intent_label, entities, prompt):
    params = []
    query,params = construct_db_query(entities, intent_label,prompt)
    if query:
        cursor.execute(query, params)
        result = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        stats = []
        for row in result:
            stats.append(dict(zip(column_names, row)))
        print(stats)
        return stats
    else:
        return generate_openai_response(prompt)


# Main endpoint
@app.post("/chat", response_model=List[dict])
async def chat(message: Message):
    user_message = message.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # NLP processing
    entities = named_entity_recog(user_message)
    intent_idx, probs = predict_intent(user_message)
    intent_label = le.inverse_transform([intent_idx])[0]
    print(intent_label)
    response = generate_response(intent_label, entities, user_message)

    chat_history.append({"sender": "user", "message": user_message})

    ai_response = {
        "sender": "ai",
        "message": str(response)
    }

    chat_history.append(ai_response)
    time.sleep(0.5)
    return chat_history

cursor = setup_database()
# Run app
if __name__ == "__main__":
    import os
    os.system("uvicorn main:app --reload")
