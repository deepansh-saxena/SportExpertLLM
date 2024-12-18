#Sport Expert LLM 
from typing import Union
from fastapi import FastAPI 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import chromadb
import os

app = FastAPI()


@app.get("/")

def sport_expert(user_prompt: str):


    chroma_client = chromadb.Client()
   
    collection = chroma_client.get_or_create_collection(name="sports")

    collection.add(
        documents=[
            '''Mysteryton: The Ultimate Game of Strategy and Agility
            Overview: Mysteryton is a dynamic and engaging sport that combines elements of strategy, agility, and teamwork. Played on a hexagonal field, Mysteryton is known for its unpredictable twists and the mental acuity required to excel. The game has gained popularity for its unique blend of physical and intellectual challenges, making it a favorite among enthusiasts of both athletics and puzzles.
            The Field: The playing field is a large hexagonal arena, divided into smaller hexagonal zones. Each zone has different properties that can affect the players' movements and the trajectory of the ball. Some zones may boost speed, while others may slow players down or alter the ball's path.
            Teams: Two teams, each consisting of six players. Positions include Strikers, Defenders, and a Strategist. The Strategist plays a crucial role, as they can change the properties of zones during the game using 'Mystery Cards.'
            Equipment: A lightweight, bouncy ball called the 'Mystery Orb.' Special shoes with enhanced grip and flexibility for navigating the varying terrain. Mystery Cards that contain power-ups, zone modifiers, or traps.
            Objective: The primary objective is to score points by getting the Mystery Orb into the opposing team's goal. Goals are positioned at opposite ends of the field. Players must navigate the shifting terrain and use their Mystery Cards strategically.
            Gameplay: 1. Kick-off: The game begins with a kick-off from the center of the hexagonal field. 2. Movement and Passing: Players pass the Mystery Orb among teammates while avoiding opponents and navigating the zones' effects. 3. Zone Activation: The Strategist can activate a Mystery Card to modify a zone, creating opportunities or obstacles. 4. Scoring: A goal is scored by successfully getting the Mystery Orb into the opposing team's goal. 5. Defense: Defenders work to intercept passes, block shots, and use their own Mystery Cards to hinder the opposing team.
            Scoring System: Each goal is worth 3 points. Additional points (1-2 points) can be earned through special achievements like trick shots, multi-pass plays, or successful traps.
            Match Duration: A standard match consists of three 15-minute periods. The team with the most points at the end of the match wins. In case of a tie, an overtime period is played, where the first team to score wins (sudden death).
            Unique Elements: Mystery Cards: These cards add an element of surprise and strategy. Each team gets a set number of cards per match, which can be used at any time. Hex Zones: The varying properties of hexagonal zones make each game unpredictable and require quick adaptation.
            Popular Tactics: Zone Control: Dominating specific zones to gain a strategic advantage. Card Timing: Using Mystery Cards at critical moments to turn the tide of the game. Team Coordination: Seamless passing and movement to outmaneuver opponents.
            Mysteryton is celebrated for its fast-paced action, the need for quick thinking, and the ability to adapt to constantly changing conditions. It's a sport that challenges both the mind and body, providing excitement for players and spectators alike.''',
            '''Mysterynis: The Enigmatic Racquet Sport
            Overview: Mysterynis is a thrilling racquet sport that blends elements of tennis, badminton, and puzzle-solving. Played on a court with ever-changing dimensions and mysterious elements, Mysterynis requires quick reflexes, strategic thinking, and adaptability. The sport has captivated audiences with its combination of physical prowess and mental challenge, making it a favorite among those who enjoy dynamic and unpredictable gameplay.
            The Court: The playing area is a rectangular court divided into two halves by a net, similar to tennis. However, the court's dimensions and features can change at intervals, introducing new challenges. Hidden zones with unique properties and obstacles can appear, affecting gameplay and requiring players to constantly adjust their strategies.
            Teams: Mysterynis can be played in singles (one player per side) or doubles (two players per side). Each player or team must work together to anticipate changes, solve on-court puzzles, and outmaneuver their opponents.
            Equipment: A specialized racquet designed for both power and precision. A lightweight, aerodynamic shuttlecock known as the 'Mystery Shuttle.' Mystery Cards that players can use to influence the court or game conditions.
            Objective: The main objective is to score points by hitting the Mystery Shuttle over the net and into the opponent's court in such a way that they are unable to return it. Players must also solve on-court puzzles and navigate obstacles that appear during the match.
            Gameplay: 1. Serve: The game starts with a serve, similar to tennis or badminton. 2. Rally: Players hit the Mystery Shuttle back and forth over the net, aiming to outplay their opponents and score points. 3. Court Changes: At random intervals, the court may change dimensions, and hidden zones or obstacles may appear, altering the dynamics of the game. 4. Mystery Cards: Players can use Mystery Cards to modify the court, introduce new challenges, or gain advantages.
            Scoring System: Points are scored when the opposing player or team fails to return the Mystery Shuttle, hits it out of bounds, or makes a fault. Each successful play earns the serving player or team one point. The first player or team to reach 21 points, with a lead of at least 2 points, wins the game. Matches are typically best of three games.
            Match Duration: Matches consist of three games, with each game played to 21 points. The player or team that wins two out of three games wins the match. Games can vary in length depending on the number of rallies and court changes.
            Unique Elements: Court Changes: The court's dimensions and features can change at random intervals, introducing new strategic elements. Mystery Cards: Players receive a limited number of Mystery Cards per match, which they can use to affect gameplay, such as altering court conditions or creating temporary obstacles. Puzzle Zones: Certain areas of the court may present puzzles that players must solve to gain an advantage or avoid penalties.
            Popular Tactics: Adaptability: Players must quickly adapt to changing court conditions and unexpected challenges. Strategic Card Use: Using Mystery Cards at optimal moments to gain an edge over opponents. Coordination: In doubles, effective communication and teamwork are crucial to handle the dynamic nature of the game.
            Mysterynis combines the fast-paced action of traditional racquet sports with the excitement of constantly evolving challenges. It demands not only athletic skill but also quick thinking and adaptability, making it an exhilarating sport for both players and spectators.'''
        ],
        ids=['id1', 'id2']
    )

        # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not found in environment variables."}
    
    llm = ChatOpenAI() #Insert API Key here

    while True:
        user_query = user_prompt
        user_query_lower = user_query.lower()
        if 'tennis' in user_query_lower:
            return 'I do not have any information regarding the sport called "Tennis"'
        elif 'badminton' in user_query_lower:
            response = llm.invoke(user_query)
            return response.content
        elif 'mysteryton' in user_query_lower:
            results = collection.query(
                query_texts=[user_query],
                n_results=1
            )
            results_text = results['documents'][0][0]
            prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                    <context>
                    {context}
                    </context>
                    Question: {input}""")
            chain = prompt | llm
            response = chain.invoke({
            "input": user_query,
            "context": results_text
            })
            return response.content
        else:
            return "Cannot answer this question"
        
