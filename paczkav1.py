import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from itertools import permutations
import random

# === Packing logic from your code ===

class Product:
    def __init__(self, width, height, depth, name=""):
        self.original_dims = (width, height, depth)
        self.name = name

    def get_orientations(self):
        return set(permutations(self.original_dims))

class PackedProduct:
    def __init__(self, position, dimensions, name):
        self.position = position
        self.dimensions = dimensions
        self.name = name

def is_collision(pos, dims, placed):
    x1, y1, z1 = pos
    w1, d1, h1 = dims
    for p in placed:
        x2, y2, z2 = p.position
        w2, d2, h2 = p.dimensions
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + d2 and y1 + d1 > y2 and
            z1 < z2 + h2 and z1 + h1 > z2):
            return True
    return False

def find_blb_position(product, placed, box_limit):
    best_pos = None
    best_volume = None
    for dims in product.get_orientations():
        w, h, d = dims
        candidate_positions = [(0,0,0)]
        for p in placed:
            px, py, pz = p.position
            pw, pd, ph = p.dimensions
            candidate_positions.extend([
                (px+pw, py, pz),
                (px, py+pd, pz),
                (px, py, pz+ph)
            ])
        for pos in candidate_positions:
            x, y, z = pos
            if x+w <= box_limit[0] and y+d <= box_limit[1] and z+h <= box_limit[2]:
                if not is_collision(pos, (w,d,h), placed):
                    vol = max(x+w, box_limit[0])*max(y+d, box_limit[1])*max(z+h, box_limit[2])
                    if best_volume is None or vol < best_volume:
                        best_volume = vol
                        best_pos = (pos, (w,d,h))
    if best_pos:
        return best_pos
    return None, None

def pack_products(products, box_limit):
    n = len(products)
    if n <= 7:
        perms = list(permutations(products))
    else:
        perms = [random.sample(products, n) for _ in range(1000)]

    best_layout = None
    best_box = None

    for order in perms:
        placed = []
        max_x = max_y = max_z = 0
        fits = True
        for p in order:
            pos, dims = find_blb_position(p, placed, box_limit)
            if pos is None:
                fits = False
                break
            placed.append(PackedProduct(pos,dims,p.name))
            max_x = max(max_x,pos[0]+dims[0])
            max_y = max(max_y,pos[1]+dims[1])
            max_z = max(max_z,pos[2]+dims[2])
        if fits:
            final_box = (max_x,max_y,max_z)
            if best_box is None or (final_box[0]*final_box[1]*final_box[2] < best_box[0]*best_box[1]*best_box[2]):
                best_box = final_box
                best_layout = placed

    return best_box, best_layout

def cuboid_data(pos, size):
    x, y, z = pos
    dx, dy, dz = size
    # Vertices of the cuboid
    return np.array([
        [x, y, z],
        [x+dx, y, z],
        [x+dx, y+dy, z],
        [x, y+dy, z],
        [x, y, z+dz],
        [x+dx, y, z+dz],
        [x+dx, y+dy, z+dz],
        [x, y+dy, z+dz]
    ])

def cuboid_faces(verts):
    return [
        [verts[j] for j in [0,1,2,3]],
        [verts[j] for j in [4,5,6,7]],
        [verts[j] for j in [0,1,5,4]],
        [verts[j] for j in [2,3,7,6]],
        [verts[j] for j in [1,2,6,5]],
        [verts[j] for j in [4,7,3,0]]
    ]

# === Dash UI ===

app = dash.Dash(__name__)
app.title = "3D Packing Online"

app.layout = html.Div([
    html.H2("Pakowanie produktów 3D (Online)"),
    html.Div([
        html.H4("Dodaj produkt"),
        html.Div([
            "Szerokość: ", dcc.Input(id='w', type='number', style={'width': '80px'}),
            "Wysokość: ", dcc.Input(id='h', type='number', style={'width': '80px'}),
            "Głębokość: ", dcc.Input(id='d', type='number', style={'width': '80px'}),
            html.Button('Dodaj', id='add', n_clicks=0),
        ]),
        html.Br(),
        html.H4("Produkty"),
        html.Ul(id='product-list'),
        html.Br(),
        html.H4("Maksymalne wymiary pudełka (X Y Z):"),
        dcc.Input(id='boxdims', placeholder='np. 30 20 10', style={'width': '170px'}),
        html.Button('Pakuj produkty', id='pack', n_clicks=0),
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '30%'}),
    html.Div([
        dcc.Graph(id='plot3d', style={'height': '500px'}),
        html.Div(id='summary', style={'marginTop': 20, 'fontSize': 18}),
    ], style={'display': 'inline-block', 'width': '65%'}),
    dcc.Store(id='products', data=[]),
])

@app.callback(
    Output('products', 'data'),
    Output('product-list', 'children'),
    Input('add', 'n_clicks'),
    State('w', 'value'),
    State('h', 'value'),
    State('d', 'value'),
    State('products', 'data'),
    prevent_initial_call=True
)
def add_product(n_clicks, w, h, d, products):
    if w and h and d:
        name = f"P{len(products)+1}"
        products.append({'w': w, 'h': h, 'd': d, 'name': name})
    items = [html.Li(f"{p['name']}: {p['w']} x {p['h']} x {p['d']}") for p in products]
    return products, items

@app.callback(
    Output('plot3d', 'figure'),
    Output('summary', 'children'),
    Input('pack', 'n_clicks'),
    State('products', 'data'),
    State('boxdims', 'value'),
    prevent_initial_call=True
)
def pack_and_plot(n_clicks, products, boxdims):
    if not products:
        return go.Figure(), "Brak produktów!"
    try:
        box_limit = tuple(map(float, boxdims.strip().split()))
        if len(box_limit) != 3:
            raise ValueError
    except:
        return go.Figure(), "Nieprawidłowe wymiary pudełka!"

    product_objs = [Product(p['w'], p['h'], p['d'], p['name']) for p in products]
    box_size, layout = pack_products(product_objs, box_limit)
    if layout is None:
        return go.Figure(), "Nie udało się zmieścić produktów w zadanym pudełku!"

    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    for idx, p in enumerate(layout):
        verts = cuboid_data(p.position, p.dimensions)
        faces = cuboid_faces(verts)
        for face in faces:
            x = [vertex[0] for vertex in face] + [face[0][0]]
            y = [vertex[1] for vertex in face] + [face[0][1]]
            z = [vertex[2] for vertex in face] + [face[0][2]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=colors[idx % len(colors)], width=5),
                showlegend=False
            ))
        cx = p.position[0] + p.dimensions[0]/2
        cy = p.position[1] + p.dimensions[1]/2
        cz = p.position[2] + p.dimensions[2]/2
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            text=[p.name],
            mode='text',
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X [cm]", range=[0, box_size[0]]),
            yaxis=dict(title="Y [cm]", range=[0, box_size[1]]),
            zaxis=dict(title="Z [cm]", range=[0, box_size[2]])
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )

    V_box = box_size[0]*box_size[1]*box_size[2]
    V_products = sum(p.dimensions[0]*p.dimensions[1]*p.dimensions[2] for p in layout)
    filled_percent = (V_products/V_box)*100
    empty_percent = 100-filled_percent

    summary_text = (
        f"Pudełko: {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} cm\n"
        f"Objętość pudełka: {V_box:.2f} cm³\n"
        f"Objętość produktów: {V_products:.2f} cm³\n"
        f"Zajętość: {filled_percent:.2f}%\n"
        f"Pusta przestrzeń: {empty_percent:.2f}%"
    )
    return fig, summary_text

if __name__ == "__main__":

    app.run_server(debug=True)
