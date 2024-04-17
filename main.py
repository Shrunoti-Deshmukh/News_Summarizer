import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
nltk.download('punkt')

# Load the fine-tuned T5 model and tokenizer from Hugging Face
model_name = "Shrunoti09/T5_News"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


st.set_page_config(page_title='RapidRecap: Summarised News📰 Portal', page_icon='./images/newspaper.ico')


def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_top_news():
    site = 'https://news.google.com/news/rss'
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_category_news(topic):
    site = 'https://news.google.com/news/rss/headlines/section/topic/{}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open('./images/no_image.jpg')
        st.image(image, use_column_width=True)


def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        # st.markdown(f"({c})[ {news.title.text}]({news.link.text})")
        st.write('**({}) {}**'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        fetch_news_poster(news_data.top_image)
        with st.expander(news.title.text):
            st.markdown(
                '''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
        st.success("Published Date: " + news.pubDate.text)
        if c >= news_quantity:
            break


# Function to summarize a custom URL
def summarize_url(url):
    article = Article(url)
    article.download()
    article.parse()
    article_text = article.text
    input_text = "summarize: " + article_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main function to run the Streamlit app
def run():

    st.title("RapidRecap: Summarised News📰")
    image = Image.open('./images/newspaper.png')
    st.image(image, use_column_width=False)

    # Navigation bar with two menu options
    menu = st.sidebar.selectbox('Navigation', ['Summarize News', 'Summarize Custom URL'])

    if menu == 'Summarize News':
        # Your existing news summarization functionality
        category = ['--Select--', 'Trending🔥 News', 'Favourite💜 Topics', 'Search🔍 Topic']
        cat_op = st.selectbox('Select your Category', category)
        if cat_op == category[0]:
            st.warning('Please select Type!!')
        elif cat_op == category[1]:
            st.subheader("✅ Here is the Trending🔥 news for you")
            no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
            news_list = fetch_top_news()
            display_news(news_list, no_of_news)
        elif cat_op == category[2]:
            av_topics = ['Choose Topic', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE',
                        'HEALTH']
            st.subheader("Choose your favourite Topic")
            chosen_topic = st.selectbox("Choose your favourite Topic", av_topics)
            if chosen_topic == av_topics[0]:
                st.warning("Please Choose the Topic")
            else:
                no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
                news_list = fetch_category_news(chosen_topic)
                if news_list:
                    st.subheader("✅ Here are the some {} News for you".format(chosen_topic))
                    display_news(news_list, no_of_news)
                else:
                    st.error("No News found for {}".format(chosen_topic))
        elif cat_op == category[3]:
            user_topic = st.text_input("Enter your Topic🔍")
            no_of_news = st.slider('Number of News:', min_value=5, max_value=15, step=1)

            if st.button("Search") and user_topic != '':
                user_topic_pr = user_topic.replace(' ', '')
                news_list = fetch_news_search_topic(topic=user_topic_pr)
                if news_list:
                    st.subheader("✅ Here are the some {} News for you".format(user_topic.capitalize()))
                    display_news(news_list, no_of_news)
                else:
                    st.error("No News found for {}".format(user_topic))
            else:
                st.warning("Please write Topic Name to Search🔍")

    elif menu == 'Summarize Custom URL':
        st.subheader("Summarize a Custom URL")
        url = st.text_input("Enter the URL:")
        if st.button("Summarize"):
            if url:
                try:
                    summary = summarize_url(url)
                    st.success("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error("Error summarizing URL. Please check the URL or try another one.")
            else:
                st.warning("Please enter a valid URL to summarize.")

run()