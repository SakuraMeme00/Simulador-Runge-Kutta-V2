import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Software Runge-Kutta 4", layout="wide")

st.title("Método de Runge-Kutta de 4to Orden (RK4)")
st.write("Software interactivo para resolver Ecuaciones Diferenciales Ordinarias (EDO)")

# --- INTERFAZ DE ENTRADA (Cajas de texto y números) ---
st.sidebar.header("Parámetros del Problema")
ecuacion_str = st.sidebar.text_input("Ecuación: dy/dx = ", value="x - y")
x0 = st.sidebar.number_input("Valor inicial de x (x0):", value=0.0)
y0 = st.sidebar.number_input("Valor inicial de y (y0):", value=2.0)
h = st.sidebar.number_input("Tamaño de paso (h):", value=0.1, step=0.01, format="%.3f")
x_final = st.sidebar.number_input("Valor final de x:", value=1.0)

# Botón para ejecutar
if st.sidebar.button("Calcular Método RK4"):
    
    # 1. Interpretar la ecuación matemáticamente
    x_sym, y_sym = sp.symbols('x y')
    try:
        expr = sp.sympify(ecuacion_str)
        f = sp.lambdify((x_sym, y_sym), expr, 'numpy')
        
        # 2. Inicializar variables
        x = x0
        y = y0
        iteracion = 0
        
        # Listas para guardar los datos
        datos_tabla = []
        x_vals = [x0]
        y_vals = [y0]
        
        # Fila inicial de la tabla (sin Ks)
        datos_tabla.append({
            "Iteración": iteracion, "x": round(x, 4), "y": round(y, 6), 
            "k1": "-", "k2": "-", "k3": "-", "k4": "-"
        })
        
        # 3. Bucle principal de Runge-Kutta
        while x < x_final - (h * 0.1):
            k1 = f(x, y)
            k2 = f(x + h/2, y + k1 * h/2)
            k3 = f(x + h/2, y + k2 * h/2)
            k4 = f(x + h, y + k3 * h)
            
            y_next = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
            x = x + h
            iteracion += 1
            y = y_next
            
            x_vals.append(x)
            y_vals.append(y)
            
            # Guardar fila de la iteración actual
            datos_tabla.append({
                "Iteración": iteracion, 
                "x": round(x, 4), "y": round(y, 6), 
                "k1": round(k1, 4), "k2": round(k2, 4), 
                "k3": round(k3, 4), "k4": round(k4, 4)
            })
            
        # --- MOSTRAR RESULTADOS EN LA PÁGINA WEB ---
        
        # Mensaje de éxito
        st.success(f"Cálculo completado. Resultado final: En **x = {x:.4f}**, el valor es **y = {y:.6f}**")
        
        # Crear dos columnas: una para la tabla y otra para la gráfica
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tabla de Iteraciones")
            # Convertimos la lista de datos en un DataFrame de Pandas para que se vea bonito
            df = pd.DataFrame(datos_tabla)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with col2:
            st.subheader("Gráfica de la Solución")
            # Generar gráfica con Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_vals, y_vals, 'bo-', linewidth=2, markersize=5, label='Aproximación RK4')
            ax.set_title(f"Solución de $dy/dx = {ecuacion_str}$")
            ax.set_xlabel("Eje X")
            ax.set_ylabel("Eje Y")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Enviar la gráfica a Streamlit
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error en la ecuación ingresada. Verifica la sintaxis (ej: usar ** para potencias). Detalle: {e}")