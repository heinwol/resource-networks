@curry
def plot_functions(range_, f1, *fs):
    fig, ax = plt.subplots()
    x = np.linspace(range_[0], range_[1], 200)
    for i, f in enumerate(itertools.chain([f1], fs)):
        y = lmap(f, x)
        ax.plot(x, y, label=f'function {i}')
    ax.legend()

@curry
def nth_pow(f, n):
    def inner(x):
        for _ in range(n):
            x = f(x)
        return x
    return inner

add1, add2 = map(curry(operator.add), [1, 2])

def make_resource_function(limitation, transition):
    def inner(x):
        if x > limitation:
            return transition
        else:
            return transition/limitation * x
    return inner

addf = lambda f1, f2: lambda x: f1(x) + f2(x)

f1 = make_resource_function(5, 7)
f2 = make_resource_function(3, 8)
f3 = make_resource_function(8, 9)


plot_here = plot_functions((0, 10))
plot_here(f1, f2)


# В композиции с собой в пределе получим константу
# plot_here(nth_pow(f1, 200))


# plot_functions((0, 10), compose(addf(f1, f2), f3), addf(compose(f1, f3), compose(f2, f3)))
# Свойства дистрибутивности не выполняются:
# plot_functions((0, 10), compose(f1, addf(f2, f3)), addf(compose(f1, f2), compose(f1, f3)))


p1, p2 = resource_piecewise(1, 4), resource_piecewise(3, 5)
comp = pw_compose(p1, p2)
comp

plot_here = plot_functions((-4, 7))

# f1_ = PiecewiseLinearFunction([1, 5, 6], [(1, 0), (3, 5), (0, 1), (-2, 4)])
# f2_ = PiecewiseLinearFunction([2, 5], [(4, 1), (-2, 8), (3, 0)])

# f1_ = PiecewiseLinearFunction([3, 4], [(1, 0), (2, 0), (3, 0)])
# f2_ = PiecewiseLinearFunction([2], [(4, 10), (1, 0)])

f1_ = PiecewiseLinearFunction([-2, 4], [(0, 0), (1, 2), (0, 0)])
f2_ = PiecewiseLinearFunction([0], [(0, 0), (-1, 0)])

comp12 = pw_compose(f1_, f2_)
print(comp12)
plot_here(comp12, compose(f1_, f2_))



#! heavily deprecated

def weights_to_labels(G, prop='weight'):
    G_ = G.copy()
    for (u, v, data) in G_.edges(data=True):
        data['label'] = data.get(prop,'')
    return G_

def display_graph(G, prop_show='weight'):
    display(Image(nx.nx_pydot.to_pydot(weights_to_labels(G, prop_show)).create_png()))
    
def random_weigthed_graph(size=6, density=0.3, seed=None):
    seed = seed if seed is not None else np.random.randint(10*15)
    G = nx.fast_gnp_random_graph(6, 0.3, seed=seed, directed=True)
    for (u, v, data) in G.edges(data=True):
        G.edges[u,v]['weight'] = np.random.randint(1,10)
    return G

# G = random_weigthed_graph()
# display_graph(G)
# nx.adjacency_matrix(G).toarray()


# # print(nx.nx_pydot.to_pydot(G).create_dot().decode('utf-8'))
# G_pd = nx.nx_pydot.to_pydot(G)
# # G_pd.set('fontsize', 4)
# G_pd.get_attributes()
# print(G_pd.create_dot().decode('utf-8'))
# # G_pd.create_dot()

G = nx.DiGraph()
G.add_edge('dsd', 1, weight=2, label='43')

G.graph['node'] = {'fontsize': 10}
# G.graph['edge'] = {'fontsize': 10}
# G.graph['rankdir'] = 'LR'

# G.remove_edge(0, 1)
# first(G.edges(data=True))[2].get('weight')
# G.nodes['dsd']['penwidth'] = 4
# G.nodes['dsd']['label'] = """<
# <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
#   <TR><TD ROWSPAN="3" BGCOLOR="yellow">class</TD></TR>
#   <TR><TD PORT="here" BGCOLOR="lightblue">qualifier</TD></TR>
# </TABLE>>"""
G.nodes['dsd']['shape'] = 'plaintext'
G.nodes['dsd']['label'] = f"""<<table>
                          <tr><td>{10}</td></tr>
                          <tr><td bgcolor='#00CC11'>B</td></tr>
                       </table>>
"""

G.nodes[1]['tooltip'] = 'lalalala'

G.nodes[1]['shape'] = 'circle'
G.nodes[1]['style'] = 'filled'
G.nodes[1]['fillcolor'] = '#f0fff4'
G.nodes[1]['fixedsize'] = True
# G.nodes[1]['fontsize'] = 10
G.nodes[1]['width'] = 0.35
G.nodes[1]['label'] = 123

G.edges['dsd', 1]['penwidth'] = 0.3
G.edges['dsd', 1]['arrowsize'] = 0.7
G.edges['dsd', 1]['fontsize'] = 10

# print(nx.nx_pydot.to_pydot(G).create_dot().decode('utf-8'))
SVG(nx.nx_pydot.to_pydot(G).create_svg())

#'color': 'transparent',
G.add_nodes_from([('lala', {'label': '', 'tooltip': ''})])
SVG(nx.nx_pydot.to_pydot(G).create_svg())

nx.nx_pydot.pydot_layout(G, prog='dot')

nx.draw(nx.fast_gnp_random_graph(8,0.2, directed=True))

G_ = nx.nx_pydot.to_pydot(G)
G_.set('rankdir', 'LR')
SVG(G_.create_svg())

play = widgets.Play(
    value=50,
    min=0,
    max=100,
    step=1,
    interval=500,
    description="Press play",
    disabled=False
)
slider = widgets.IntSlider()
widgets.jslink((play, 'value'), (slider, 'value'))
widgets.HBox([play, slider])
