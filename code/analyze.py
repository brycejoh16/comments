import json

import matplotlib.pyplot as plt
import openai
import pandas as pd
from keys import open_ai_key,open_ai_organization
from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm
import datetime



client = OpenAI(api_key=open_ai_key,organization=open_ai_organization)


def get_embedding(text, model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def obtain_embeddigns(category,video_id, model):
    comment_csv = f'output/search/{category}/{video_id}/comments.csv'
    df = pd.read_csv(comment_csv)
    df['Comment'] = df['Comment'].apply(lambda  x: str(x))
    df = df.set_index('CommentID')
    df[model]  = df['Comment'].apply(lambda x: get_embedding(str(x),model))
    df.to_csv(comment_csv)

def do_kmeans(matrix,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    return labels

def weighted_mean(values, weights):
    return np.average(values, weights=weights)

def analyze_embeddings(category,video_id, model,summary_model,nb_clusters=5):
    comment_csv= f'output/search/{category}/{video_id}/comments.csv'
    stats_yaml= f'output/search/{category}/{video_id}/stats.yaml'

    with open(stats_yaml, 'r') as file:
        stats_dict = yaml.safe_load(file)

    df = pd.read_csv(comment_csv)
    df['Comment'] = df['Comment'].apply(lambda x:str(x))
    df=df.set_index('CommentID')
    # df[model]  = df['Comment'].apply(lambda x: get_embedding(x,model))

    df[model] = df[model].apply(lambda x:np.array(eval(x)))

    labels =do_kmeans(np.array(list(df[model])),nb_clusters)

    pca = PCA(n_components=2)
    X_new = pca.fit_transform(list(df[model]))



    fig, ax = plt.subplots(1, 1,figsize=(12,12))

    colors =["purple", "green", "red", "blue", "pink",'orangered','cyan','lime','navy','tan']
    stats_dict['clusters']={}

    AXs= [ ]
    for i, color in enumerate(colors[:nb_clusters]):
        stats_dict[color]=[]
        xs = np.array(X_new[:,0])[labels == i]
        ys = np.array(X_new[:,1])[labels == i]

        stats_dict['clusters'][f"{color}"] = {'size':xs.shape[0]}
        color_ax=ax.scatter(xs, ys, color=color, alpha=0.3)
        AXs.append(color_ax)
        # weight by the number of likes based on how popular each idea is ...
        # even though this is a flawed way of looking at things.

        avg_x = weighted_mean(xs,df['LikeCount'][labels == i]+1)
        avg_y =  weighted_mean(ys,df['LikeCount'][labels == i]+1)
        ax.scatter(avg_x, avg_y, marker="x", color=color, s=100)

        nb_comments= 5
        lowest_indices = np.argsort(np.sqrt((avg_y-X_new[:,1])**2 + (avg_x-X_new[:,0])**2))[:nb_comments]
        for min_index in lowest_indices:
            stats_dict[color].append( df.iloc[min_index]['Comment'])

        #### now we need to analyze each cluster
    stats_dict["GPT-Summary"]={}

    Subjects = []
    for i, color in tqdm(enumerate(colors[:nb_clusters])):
        comments_cluster= stats_dict[color]
        comments_not_cluster=[]
        for color_not_in_cluster in colors[:nb_clusters]:
            if color_not_in_cluster!=color:
                comments_not_cluster+=stats_dict[color_not_in_cluster]
        comments_not_cluster = '-' + '\n -'.join(list(comments_not_cluster))
        comments_cluster= '-' + '\n -'.join(list(comments_cluster))
        messages = [
            {"role": "user",
             "content": f'How are these comments Comments:\n"""\n{comments_cluster}\n""" different from these comments '
                        f' Comments:\n"""\n{comments_not_cluster}\n""" '
                        f'in a YouTube video titled {stats_dict["VideoTitle"]} by {stats_dict["ChannelName"]} After comparing the difference '
                        f'please give 1 sentence summary of what the first set of comments are about. Please omit the usage of "The first set of comments are primarily focused on" '
                        f'in the 1 sentence summary.'
                        f'After that please output a single line Subject:" " and place the subject of the 1 sentence summaryFor reactions that are simple please charactize these as "simple reactions".'
                        f' \n'
             }]


        #                 f''
        #                 f'What do the following YouTube comments to the video titled {stats_dict["VideoTitle"]} by {stats_dict["ChannelName"]} '
        #                 f'have in commmon  Similar Comments:\n"""\n{comments_cluster}\n"""'
        #                 f'\n\n Contrasting Comments:\n"""\n{comments_not_cluster}\n"""'
        #                 f'\n\nTheme:'}
        # ]

        response = client.chat.completions.create(
            model=summary_model,
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

        response_message = response.choices[0].message.content

        subject = response_message.split('Subject:')[-1]

        Subjects.append(subject)
        stats_dict["GPT-Summary"][color] = response_message




    for color_ax,subject,color in zip(AXs,Subjects,colors):
        color_ax.set_label(f"{subject}:{stats_dict['clusters'][color]['size']}")


    ax.legend()
    # ax.scatter(X_new[:,0],X_new[:,1],alpha=0.5,s=0.5)
    ax.set_xlabel('Dimension 1',fontname='Times New Roman')
    ax.set_ylabel('Dimension 2',fontname='Times New Roman')
    # ax.set_xticklabels(fontname='Times New Roman')
    # ax.set_xticks(fontsize=10, fontname='Times New Roman')
    ax.set_title(f'Channel : {stats_dict["ChannelName"]} \n Title: {stats_dict["VideoTitle"]}',fontname='Times New Roman')
    fig.tight_layout()
    fig.savefig(f'output/search/{category}/{video_id}/pca_{model}.png')


    df[f'{model}-pca']=list(X_new)
    df[f'{model}-pca'] =df[f'{model}-pca'].apply(lambda x:list(x))



    stats_dict['explained_variance_ratio_']=  str(pca.explained_variance_ratio_)
    stats_dict['singular_values_']=str(pca.singular_values_)


    with open(stats_yaml, 'w') as yaml_file:
        yaml.dump(stats_dict, yaml_file, default_flow_style=False)


def analyze_all():
    # topics = ['climate_change', 'AI', 'GMO']
    topics = ['AlphaFold_3']
    # ids = ['-Vm_gabtIA8','0l4N1wjoKAI','2njn71TqkjA',]
    # ids=['3OTxdgMJsnw','8ONGuJCIkpQ','97nEBjiQI1M']
    # ids = ['1DrltazeqTA','2eWvvLBrrlE','2G-yUuiqIZ0','3eybu-IgeQc']
    for topic in topics:
        with open(f'output/search/{topic}/videos.json', 'r') as f:
            ids = json.load(f)

        for video_id in tqdm(ids):
            # Get the current timestamp
            try:
                timestamp = datetime.datetime.now()

                # Convert the timestamp to a string with a specific format
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                # Print the timestamp along with the message
                print(f"{timestamp_str}: time stamp")

                stats_yaml = f'output/search/{topic}/{video_id}/stats.yaml'
                with open(stats_yaml, 'r') as file:
                    stats_dict = yaml.safe_load(file)
                    if 'commentCount' in stats_dict['Statistics'].keys():
                        year , month, day = stats_dict['PublishDate'].split('-')
                        if int(year) ==2024:
                            if int(month) == 5:
                                if int(day) > 7:
                                    if int(stats_dict['Statistics']['commentCount']) < 1000:
                                        print('running : ', stats_dict['VideoTitle'], 'id', stats_dict['VideoID'], 'comment',
                                              int(stats_dict['Statistics']['commentCount']))

                                        obtain_embeddigns(topic, video_id, "text-embedding-3-large")
                                        analyze_embeddings(topic, video_id, "text-embedding-3-large", "gpt-4-turbo-2024-04-09")

                                    else:
                                        print('skipping ', stats_dict['VideoTitle'], ' too big !', 'id', stats_dict['VideoID'],
                                              'comment', int(stats_dict['Statistics']['commentCount']))
                    else:
                        print('skipping ', stats_dict['VideoTitle'], ' no comments', 'id', stats_dict['VideoID'], )

            except Exception as E:
                print("An error occurred:", E)



if __name__ == '__main__':
    #
    # ids  =['uynhvHZUOOo']
    # for id in ids:
    #     analyze_embeddings('climate_change',id,"text-embedding-3-large", "gpt-4-turbo-2024-04-09",5)
    analyze_all()





