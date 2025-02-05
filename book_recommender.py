import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch
from torch.autograd import grad
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import aiohttp
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

# ===== init =====
st.set_page_config(page_title="Book Recommender")
CACHE_DIR = "./model_cache"
MODEL_NAME = "avsolatorio/NoInstruct-small-Embedding-v0"
DATASET_PATH = './books.csv'
EMBEDDINGS_PATH = './book_embeddings.npy'

def init_session_state():
    state_keys = {
        'user_prefs': None,
        'model': None,
        'tokenizer': None,
        'dataset': None,
        'embeddings': None,
        'search_list': None,
        'pref_recommendations': None,
        'text_recommendations': None,
        'dialog_book': None,
    }
    for key, default in state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ===== load models and dataset =====
@st.cache_resource
def load_model():
    try:
        cache_path = Path(CACHE_DIR) / MODEL_NAME.replace("/", "_")
        model = AutoModel.from_pretrained(cache_path)
        tokenizer = AutoTokenizer.from_pretrained(cache_path)
        model.eval()
        return model, tokenizer

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        return df, embeddings
    except FileNotFoundError:
        st.error("Dataset files not found. Please check the dataset path.")
        st.stop()

# ========== get recommendations ===========
async def fetch_book_data(title, author):
    # dicebear cover image for fallback
    cover = f"https://api.dicebear.com/9.x/glass/svg?seed={title.replace(' ', '')}"
    wiki = f"https://en.wikipedia.org/wiki/{quote(f'{title} ({author} novel)')}"

    async with aiohttp.ClientSession() as session:
        base_params = {"format": "json", "action": "query"}
        ol_url = f"https://openlibrary.org/search.json?q={quote(f'{title} {author}')}"
        
        # get the URLs
        ol_resp, wiki_resp = await asyncio.gather(
            session.get(ol_url),
            session.get("https://en.wikipedia.org/w/api.php", params={
                **base_params,
                "list": "search",
                "srsearch": f"{title} {author} novel"
            })
        )
    
        try:
            if docs := (await ol_resp.json()).get('docs'):
                if cover_id := docs[0].get('cover_i'):
                    cover = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        except Exception:
            pass

        try:
            if search := (await wiki_resp.json()).get('query', {}).get('search'):
                page = search[0]['title']
                wiki = f"https://en.wikipedia.org/wiki/{quote(page)}"
        except Exception:
            pass

        return cover, wiki

async def get_recommendations(query_embedding, limit=6):
    # get similarity scores
    scores = np.dot(query_embedding, st.session_state.embeddings.T)
    top_indices = np.argsort(-scores)[:limit]
    
    # get most similar metadata
    books = st.session_state.dataset.iloc[top_indices]
    
    # get cover image and wikipedia link
    async with aiohttp.ClientSession() as session:
        image_data = await asyncio.gather(*(
            fetch_book_data(title, author) for title, author in books[['name', 'author']].values
        ))

    return [{
        "book_position": pos,
        "name": title,
        "author": author,
        "image_url": img,
        "preview_url": url,
    } for (title, author), (img, url), pos in zip(books[['name', 'author']].values, image_data, top_indices)]


# ======== card modal =========
@st.dialog("Book Details", width="large")
def show_book_details():
    book = st.session_state.dialog_book
    ds_row = st.session_state.dataset.loc[book['book_position']]
    img_col, _, text_col = st.columns([4, 1, 8])
    
    with img_col:
        st.image(book['image_url'], use_container_width=True)
    
    with text_col:
        st.markdown(f"# {book['name']}")
        st.markdown(f"**By {book['author']}**  |  *{ds_row['genre'].title()}*")
        
        with st.expander("Summary"):
            st.write(ds_row['summary'])
        
        st.link_button("Learn More", book['preview_url'], 
                      use_container_width=True, type="primary")

# Preference Searchbar
def pref_search():
    selected = st.session_state.search_select
    st.session_state.search_list.remove(selected)
    name, author = selected.split(", ", 1)
    
    # Find book position
    ds = st.session_state.dataset
    book_mask = (ds['name'] == name) & (ds['author'] == author)
    book_position = book_mask.values.nonzero()[0][0]
    
    # Add new preference
    image, preview_url = asyncio.run(fetch_book_data(name, author))
    new_pref = {
        "book_position": book_position,
        "name": name,
        "author": author,
        "image_url": image,
        "preview_url": preview_url,
    }
    st.session_state.user_prefs = (st.session_state.user_prefs or []) + [new_pref]


# ======== recommendation explanation ========
def combine_subword_tokens(tokens, scores):
   words = []
   curr_word, curr_scores = [], []
   special_tokens = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
   
   i = 0
   while i < len(tokens):
       token, score = tokens[i], scores[i]
       
       # no special tokens
       if token in special_tokens:
           i += 1
           continue
       
       # hyphenated words (mostly just sci-fi)
       if i + 2 < len(tokens) and tokens[i+1] == '-':
           curr_word.extend([token, '-', tokens[i+2]])
           curr_scores.extend([score, scores[i+1], scores[i+2]])
           i += 3
           
       elif not token.startswith('##'):
           if curr_word:
               words.append((''.join(curr_word), np.mean(curr_scores)))
               curr_word, curr_scores = [], []
           curr_word.append(token)
           curr_scores.append(score)
           i += 1
           
       else:
           curr_word.append(token[2:])
           curr_scores.append(score)
           i += 1
   
   # final word
   if curr_word:
       words.append((''.join(curr_word), np.mean(curr_scores)))
       
   return words

def text_explanation(text_input: str, n_steps: int = 50) -> None:
    # Setup model and targets
    positions = [book['book_position'] for book in st.session_state.text_recommendations]
    model = st.session_state.model
    device = next(model.parameters()).device
    tokenizer = st.session_state.tokenizer

    # Prepare target embeddings
    target_emb = torch.tensor(st.session_state.embeddings[positions], 
                            device=device, dtype=torch.float16)
    target_emb = F.normalize(target_emb, p=2, dim=1).to(torch.float16)

    enc = tokenizer(text_input, padding=True, truncation=True,
                    return_tensors='pt', max_length=512)
    inp_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)

    # embeddings
    with torch.no_grad():
        init_emb = model.embeddings.word_embeddings(inp_ids)
    base_emb = torch.zeros_like(init_emb, dtype=torch.float16)
    int_grads = torch.zeros_like(init_emb, dtype=torch.float16)

    def get_embedding(token_emb, attn_mask):
        """Get normalized CLS embedding"""
        out = model(inputs_embeds=token_emb.to(torch.float16),
                    attention_mask=attn_mask.to(torch.float16),
                    return_dict=True)
        return F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)

    # Calculate integrated gradients
    for alpha in torch.linspace(0, 1, n_steps):
        scaled_emb = base_emb + alpha * (init_emb - base_emb)
        scaled_emb = scaled_emb.to(torch.float16)
        scaled_emb.requires_grad_(True)
        
        with torch.set_grad_enabled(True):
            curr_emb = get_embedding(scaled_emb, attn_mask).to(torch.float16)
            sim = torch.matmul(curr_emb, target_emb.T).mean().to(torch.float16)
            
            grads = grad(sim, scaled_emb)[0]           
            if torch.isnan(grads).any():
                st.write(f"NaN in gradients at alpha={alpha}, sim={sim.item()}")
                continue
                
            int_grads += grads.detach() / n_steps
   
    # token importance
    attr = (init_emb - base_emb) * int_grads
    importance = (attr.sum(dim=2) * attn_mask).squeeze(0)
    scores = importance.detach().cpu().numpy()

    # scores
    tokens = tokenizer.convert_ids_to_tokens(inp_ids[0])
    token_scores = [(t, float(s)) for t, s in zip(tokens, scores) 
                    if t not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']]
    token_scores = combine_subword_tokens(tokens, scores)
    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    st.markdown("### Key words driving recommendations")
    df = pd.DataFrame({
        'Word': [t for t, _ in token_scores[:5]],
        'Impact': [s for _, s in token_scores[:5]]
    })
    st.bar_chart(df.set_index('Word')['Impact'], use_container_width=True, 
                horizontal=True)

def plot_book_similarities():
    # Get positions and embeddings
    user_pos = [b['book_position'] for b in st.session_state.user_prefs]
    user_emb = [st.session_state.embeddings[p] for p in user_pos]
    rec_emb = [st.session_state.embeddings[b['book_position']] 
               for b in st.session_state.pref_recommendations]
    titles = [st.session_state.dataset.iloc[p]['name'] for p in user_pos]

    # Calculate similarities
    avg_rec = np.mean(rec_emb, axis=0).reshape(1, -1)
    sim_values = [cosine_similarity(emb.reshape(1, -1), avg_rec)[0][0] 
                 for emb in user_emb]
    
    # softmax (just helps with visualization)
    sim_exp = np.exp(sim_values)
    sim_probs = sim_exp / sim_exp.sum()
    sims = {title: prob for title, prob in zip(titles, sim_probs)}
    
    # plot
    st.markdown("### Key preferences driving recommendations")
    st.bar_chart(dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)), 
                horizontal=True)

# ====== main =====
def main():
    # init
    init_session_state()
    with st.spinner("loading model"):
        if st.session_state.model is None:
            st.session_state.model, st.session_state.tokenizer = load_model()
    with st.spinner("loading dataset"):
        if st.session_state.dataset is None:
            st.session_state.dataset, st.session_state.embeddings = load_dataset()
            st.session_state.search_list = [
                f"{row['name']}, {row['author']}" 
                for _, row in st.session_state.dataset.iterrows()
            ]

    st.title(":books: Book Recommender")       

    tabs = st.tabs([":pencil2:  Describe A Book", ":page_facing_up:  From Your History"])
    with tabs[0]:
        text_input = st.text_area("Your next read will have...", height=200)

        text_button_cols = st.columns(3)
        with text_button_cols[2]:
            if st.button("Get Recommendations", key="Text_Rec_Button", use_container_width=True):

                with st.spinner("Analyzing text..."):
                    with torch.no_grad():
                        inp = st.session_state.tokenizer([text_input], return_tensors="pt", padding=True, truncation=True, max_length=512)                       
                        output = st.session_state.model(**inp)

                    masked = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
                    pooled = masked.sum(1) / inp["attention_mask"].sum(-1).view(-1, 1)
                    query_embedding = F.normalize(pooled[0], p=2, dim=0).numpy()

                    recommendations = asyncio.run(get_recommendations(query_embedding))
                    st.session_state.text_recommendations = recommendations


        if st.session_state.text_recommendations:
            st.markdown("#### Recommended Books")
            cols = st.columns(3)
            for idx, rec in enumerate(st.session_state.text_recommendations):
                with cols[idx % 3]:
                    with st.container():
                        # Display book image
                        with st.spinner(f"Loading {rec['name']} image"):
                            st.image(rec["image_url"], use_container_width=True)

                        if st.button(f"**{rec['name']}**  \n  *{rec['author']}*", key=f"text_{rec['name']}_{rec['author']}", use_container_width=True, type="secondary", help="Click me!"):
                            st.session_state.dialog_book = rec
                            st.session_state.show_dialog = True
                            show_book_details()

            st.write("") 
            text_explanation(text_input)


    with tabs[1]:
        st.selectbox(
            "Search your liked books...",
            options=st.session_state.search_list,
            on_change=pref_search,
            key="search_select",
            index=None
        )
    
        if st.session_state.user_prefs:
            search_card_cols = st.columns(3)
            for idx, pref in enumerate(reversed(st.session_state.user_prefs[-3:])):
                with search_card_cols[idx]:
                    with st.container():
                        # Display book image
                        with st.spinner(f"Loading {pref['name']} image"):
                            st.image(pref["image_url"], use_container_width=True)

                        button_string = f"**{pref['name']}**   \n  *{pref['author']}*"
                        if st.button(button_string, use_container_width=True, key=f"Pref_Select_Button_{pref['name']}_{pref['author']}", type="secondary", help="Click me!"):
                            st.session_state.dialog_book = pref
                            show_book_details()

            search_button_cols = st.columns(3)
            with search_button_cols[2]:
                if st.button("Get Recommendations", key="Pref_Rec_Button", use_container_width=True):
                    with st.spinner("Finding recommendations..."):
                        avg_embedding = np.mean([st.session_state.embeddings[pref['book_position']] for pref in st.session_state.user_prefs[-3:]], axis=0)
                        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)   
                        recommendations = asyncio.run(get_recommendations(avg_embedding, limit=6+len(st.session_state.user_prefs)))
                        st.session_state.pref_recommendations = [
                            rec for rec in recommendations
                            if rec['name'] not in {pref['name'] for pref in st.session_state.user_prefs}
                        ]
        if st.session_state.pref_recommendations:
            st.markdown("#### Recommended Books")
            cols = st.columns(3)
            for idx, rec in enumerate(st.session_state.pref_recommendations):
                with cols[idx % 3]:
                    with st.container():
                        # Display book image
                        with st.spinner(f"Loading {rec['name']} image"):
                            st.image(rec["image_url"], use_container_width=True)

                        if st.button(f"**{rec['name']}**  \n  *{rec['author']}*", key=f"pref_{rec['name']}_{rec['author']}", use_container_width=True, type="secondary", help="Click me!"):
                            st.session_state.dialog_book = rec
                            st.session_state.show_dialog = True
                            show_book_details()
            
            st.write("")
            plot_book_similarities()

if __name__ == '__main__':
    main()

