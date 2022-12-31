# from ast import pattern
import plotly.graph_objects as go
x = ["BERT-C","TextCNN"]
ori=[97,93]
A2T = [81,79]
BAE = [74,71]
FAGA = [47,41]
BERT_A = [45,33]
CLARE = [21,14]
PWWS = [18,11]
PSO = [19,15]
FBA = [9,10]
data=[ori,A2T,BAE,FAGA,BERT_A,CLARE,PWWS,PSO,FBA]
names=['No Attack','A2T','BAE','FAGA','BERT-A','CLARE','PWWS','PSO','FBA']
pattern=['','/','\\', 'x','-','|','+','.','']
fig = go.Figure()
for i in range(len(data)):    
    fig.add_trace(go.Histogram(histfunc="sum",y=data[i], x=x,name=names[i]))
    # fig.update_traces(marker_pattern_shape=pattern[i])
fig.update_layout(
        # legend_title="Attacks",
        legend = dict(font = dict(family = "Courier", size = 45, color = "black")),
        font=dict(
            family="Courier New, monospace",
            size=55,
            color="RebeccaPurple"
        )
    )
# fig.update_layout(annotations=[
#         go.layout.Annotation(
#             showarrow=False,
#             font=dict(
#                 family="Courier New, monospace",
#                 size=30,
#                 color="#0000FF"
#             )
#         )])
fig.update_yaxes(title_text="Accuracy After Attack(%)")
fig.update_layout(xaxis = dict(tickfont = dict(size=55)))
fig.update_layout(yaxis = dict(tickfont = dict(size=55)))
fig.update_layout(
# legend_title='Attack Methods',
legend=dict(
orientation="h",
yanchor="bottom",
y=1.05,
xanchor="right",
x=1))
fig.update_layout(legend= {'itemsizing': 'constant'})
# fig.update_layout(legend=dict(yanchor="top", xanchor="left"))
# fig.update_layout(legend = dict(font = dict(family = "Courier", size = 50, color = "black")))
fig.show()

