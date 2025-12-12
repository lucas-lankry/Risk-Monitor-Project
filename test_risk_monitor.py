import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# Configuration de la page
st.set_page_config(page_title="Options Risk Monitor", layout="wide", initial_sidebar_state="expanded")

# Titre principal
st.title("Options Risk Monitor - Futures")
st.markdown("---")

# Initialiser le session state pour stocker les positions
if 'positions' not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=[
        'Ticker', 'Strike', 'Position_Size', 'Maturity', 'Settlement_Price', 'Contract_Size', 'Type'
    ])

# Initialiser les DataFrames pour chaque modèle de Greeks
if 'b76_greeks' not in st.session_state:
    st.session_state.b76_greeks = pd.DataFrame(columns=[
        'Product', 'Position_Size', 'Bid', 'Ask', 'Last_Price', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'PnL'
    ])

if 'bachelier_greeks' not in st.session_state:
    st.session_state.bachelier_greeks = pd.DataFrame(columns=[
        'Product', 'Position_Size', 'Bid', 'Ask', 'Last_Price', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'PnL'
    ])

if 'local_vol_greeks' not in st.session_state:
    st.session_state.local_vol_greeks = pd.DataFrame(columns=[
        'Product', 'Position_Size', 'Bid', 'Ask', 'Last_Price', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'PnL'
    ])

if 'heston_greeks' not in st.session_state:
    st.session_state.heston_greeks = pd.DataFrame(columns=[
        'Product', 'Position_Size', 'Bid', 'Ask', 'Last_Price', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'PnL'
    ])

if 'rates_data' not in st.session_state:
    st.session_state.rates_data = None

if 'show_positions' not in st.session_state:
    st.session_state.show_positions = False

# Fonction pour sauvegarder les données
def save_data():
    data = {
        'positions': st.session_state.positions.to_dict(),
        'b76_greeks': st.session_state.b76_greeks.to_dict(),
        'bachelier_greeks': st.session_state.bachelier_greeks.to_dict(),
        'local_vol_greeks': st.session_state.local_vol_greeks.to_dict(),
        'heston_greeks': st.session_state.heston_greeks.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    with open('risk_monitor_data.json', 'w') as f:
        json.dump(data, f)

# Fonction pour calculer les Greeks (placeholder - à remplacer par ton code)
def run_calculation():
    """
    Cette fonction devrait appeler ton code existant de calcul des Greeks
    """
    # TODO: Intégrer ton code de calcul des Greeks ici
    st.info("Calcul des Greeks en cours... (a implementer avec votre code)")
    # Exemple: from greeks_calculator import calculate_all_greeks
    # st.session_state.b76_greeks = calculate_b76_greeks(st.session_state.positions)
    # st.session_state.bachelier_greeks = calculate_bachelier_greeks(st.session_state.positions)
    # etc.

# Fonction pour envoyer par email
def send_email(file_path, recipient_email, sender_email, sender_password):
    """
    Fonction pour envoyer le fichier par email
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Risk Monitor Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = "Veuillez trouver ci-joint le rapport Risk Monitor."
        msg.attach(MIMEText(body, 'plain'))
        
        with open(file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
        msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True, None
    except Exception as e:
        return False, str(e)

# Sidebar pour la gestion des données
with st.sidebar:
    st.header("Configuration")
    
    mode = st.radio(
        "Mode de travail",
        ["Saisie manuelle", "Import Excel"]
    )
    
    if mode == "Import Excel":
        uploaded_file = st.file_uploader("Charger le fichier Excel", type=['xlsx', 'xls'])
        
        if uploaded_file:
            try:
                xls = pd.ExcelFile(uploaded_file)
                selected_sheet = st.selectbox("Selectionner l'onglet", xls.sheet_names)
                
                if st.button("Importer les donnees"):
                    st.session_state.positions = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    # Charger aussi les Greeks si disponibles
                    if 'B76_Greeks' in xls.sheet_names:
                        st.session_state.b76_greeks = pd.read_excel(uploaded_file, sheet_name='B76_Greeks')
                    if 'Bachelier_Greeks' in xls.sheet_names:
                        st.session_state.bachelier_greeks = pd.read_excel(uploaded_file, sheet_name='Bachelier_Greeks')
                    if 'Local_Vol_Greeks' in xls.sheet_names:
                        st.session_state.local_vol_greeks = pd.read_excel(uploaded_file, sheet_name='Local_Vol_Greeks')
                    if 'Heston_Greeks' in xls.sheet_names:
                        st.session_state.heston_greeks = pd.read_excel(uploaded_file, sheet_name='Heston_Greeks')
                    if 'US_Rates_Curve' in xls.sheet_names:
                        st.session_state.rates_data = pd.read_excel(uploaded_file, sheet_name='US_Rates_Curve')
                    
                    st.success("Donnees importees avec succes")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'import : {e}")
    
    st.markdown("---")
    
    # Actions
    st.subheader("Actions")
    
    if st.button("Actualiser", use_container_width=True):
        if not st.session_state.positions.empty:
            with st.spinner("Calcul en cours..."):
                run_calculation()
                save_data()
                st.rerun()
        else:
            st.warning("Aucune position a calculer")
    
    if st.button("Modifier ou Supprimer une position", use_container_width=True):
        st.session_state.show_positions = True
    
    st.markdown("---")
    
    # Export
    st.subheader("Export")
    
    if not st.session_state.positions.empty:
        # Export Excel
        if st.button("Export Excel", use_container_width=True):
            output_file = f"risk_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Export positions
                st.session_state.positions.to_excel(writer, sheet_name='Positions', index=False)
                
                # Export tous les modèles de Greeks
                if not st.session_state.b76_greeks.empty:
                    st.session_state.b76_greeks.to_excel(writer, sheet_name='B76_Greeks', index=False)
                
                if not st.session_state.bachelier_greeks.empty:
                    st.session_state.bachelier_greeks.to_excel(writer, sheet_name='Bachelier_Greeks', index=False)
                
                if not st.session_state.local_vol_greeks.empty:
                    st.session_state.local_vol_greeks.to_excel(writer, sheet_name='Local_Vol_Greeks', index=False)
                
                if not st.session_state.heston_greeks.empty:
                    st.session_state.heston_greeks.to_excel(writer, sheet_name='Heston_Greeks', index=False)
                
                # Export courbe des taux si disponible
                if st.session_state.rates_data is not None:
                    st.session_state.rates_data.to_excel(writer, sheet_name='US_Rates_Curve', index=False)
            
            st.success(f"Fichier exporte : {output_file}")
        
        # Export CSV
        csv = st.session_state.positions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export CSV",
            data=csv,
            file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Send by Email
        if st.button("Send by eMail", use_container_width=True):
            st.session_state.show_email_form = True
        
        if 'show_email_form' in st.session_state and st.session_state.show_email_form:
            with st.form("email_form"):
                st.markdown("### Configuration Email")
                
                sender_email = st.text_input("Votre email (expediteur)", placeholder="votre.email@gmail.com")
                sender_password = st.text_input("Mot de passe d'application Gmail", type="password", 
                                               help="Utilisez un mot de passe d'application, pas votre mot de passe Gmail principal")
                st.markdown("---")
                recipient = st.text_input("Email destinataire", placeholder="destinataire@example.com")
                
                st.info("Pour Gmail : Vous devez creer un mot de passe d'application sur https://myaccount.google.com/apppasswords")
                
                submit = st.form_submit_button("Envoyer")
                cancel = st.form_submit_button("Annuler")
                
                if cancel:
                    st.session_state.show_email_form = False
                    st.rerun()
                
                if submit and recipient and sender_email and sender_password:
                    # Créer un fichier temporaire
                    temp_file = f"temp_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                        st.session_state.positions.to_excel(writer, sheet_name='Positions', index=False)
                        if not st.session_state.b76_greeks.empty:
                            st.session_state.b76_greeks.to_excel(writer, sheet_name='B76_Greeks', index=False)
                        if not st.session_state.bachelier_greeks.empty:
                            st.session_state.bachelier_greeks.to_excel(writer, sheet_name='Bachelier_Greeks', index=False)
                        if not st.session_state.local_vol_greeks.empty:
                            st.session_state.local_vol_greeks.to_excel(writer, sheet_name='Local_Vol_Greeks', index=False)
                        if not st.session_state.heston_greeks.empty:
                            st.session_state.heston_greeks.to_excel(writer, sheet_name='Heston_Greeks', index=False)
                    
                    success, error = send_email(temp_file, recipient, sender_email, sender_password)
                    
                    if success:
                        st.success("Email envoye avec succes!")
                        os.remove(temp_file)
                        st.session_state.show_email_form = False
                        st.rerun()
                    else:
                        st.error(f"Erreur lors de l'envoi: {error}")
                        st.warning("Verifiez que vous utilisez un mot de passe d'application Gmail, pas votre mot de passe principal.")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                
                elif submit:
                    st.warning("Veuillez remplir tous les champs")

# Affichage principal
df = st.session_state.positions

# Metrics en haut
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Nombre de Positions",
            value=len(df),
        )
    
    with col2:
        if 'Position_Size' in df.columns:
            total_position = df['Position_Size'].sum()
            st.metric(
                label="Position Totale",
                value=f"{int(total_position)}",
            )
    
    with col3:
        # Calculer Delta total depuis B76_Greeks si disponible
        if not st.session_state.b76_greeks.empty and 'Delta' in st.session_state.b76_greeks.columns:
            total_delta = st.session_state.b76_greeks['Delta'].sum()
            st.metric(
                label="Delta Total",
                value=f"{total_delta:.2f}",
                delta=f"{total_delta:.2f}" if total_delta != 0 else None
            )
        else:
            st.metric(label="Delta Total", value="0.00")
    
    with col4:
        # Calculer Gamma total depuis B76_Greeks si disponible
        if not st.session_state.b76_greeks.empty and 'Gamma' in st.session_state.b76_greeks.columns:
            total_gamma = st.session_state.b76_greeks['Gamma'].sum()
            st.metric(
                label="Gamma Total",
                value=f"{total_gamma:.2f}",
                delta=f"{total_gamma:.2f}" if total_gamma != 0 else None
            )
        else:
            st.metric(label="Gamma Total", value="0.00")
    
    st.markdown("---")

# Tabs pour différentes vues
if st.session_state.show_positions:
    # Forcer l'affichage de l'onglet Positions
    if mode == "Saisie manuelle":
        selected_tab = st.radio("Navigation", ["Nouvelle Position", "Positions", "Greeks", "Courbe des Taux", "Analyse"], 
                                index=1, horizontal=True, label_visibility="collapsed")
    else:
        selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"], 
                                index=0, horizontal=True, label_visibility="collapsed")
    st.session_state.show_positions = False
else:
    if mode == "Saisie manuelle":
        selected_tab = st.radio("Navigation", ["Nouvelle Position", "Positions", "Greeks", "Courbe des Taux", "Analyse"], 
                                index=0, horizontal=True, label_visibility="collapsed")
    else:
        selected_tab = st.radio("Navigation", ["Positions", "Greeks", "Courbe des Taux", "Analyse"], 
                                index=0, horizontal=True, label_visibility="collapsed")

if mode == "Saisie manuelle" and selected_tab == "Nouvelle Position":
    st.subheader("Ajouter une nouvelle position")
    
    with st.form("new_position_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", value="CLG6C")
            strike = st.number_input("Strike", value=58.0, step=0.5)
        
        with col2:
            position_size = st.number_input("Position Size", value=10, step=1)
            maturity = st.text_input("Maturity", value="FEB 26")
        
        with col3:
            settlement_price = st.number_input("Settlement Price", value=0.0, step=0.01)
            contract_size = st.number_input("Contract Size", value=1000, step=100)
        
        with col4:
            option_type = st.selectbox("Type", ["C", "P"])
        
        submitted = st.form_submit_button("Ajouter la position", use_container_width=True)
        
        if submitted:
            new_row = {
                'Ticker': ticker,
                'Strike': strike,
                'Position_Size': position_size,
                'Maturity': maturity,
                'Settlement_Price': settlement_price,
                'Contract_Size': contract_size,
                'Type': option_type
            }
            
            # Ajouter la nouvelle ligne
            st.session_state.positions = pd.concat([
                st.session_state.positions,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            save_data()
            st.success("Position ajoutee ! Cliquez sur Actualiser pour calculer les Greeks.")
            st.rerun()

if selected_tab == "Positions":
    st.subheader("Positions actuelles")
    
    if not df.empty:
        # Filtres pour le tableau
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            if 'Type' in df.columns:
                option_type = st.multiselect(
                    "Type d'option",
                    options=df['Type'].unique(),
                    default=df['Type'].unique()
                )
                df_filtered = df[df['Type'].isin(option_type)]
            else:
                df_filtered = df
        
        with col_filter2:
            if 'Maturity' in df.columns:
                maturities = st.multiselect(
                    "Maturite",
                    options=sorted(df['Maturity'].unique()),
                    default=sorted(df['Maturity'].unique())
                )
                df_filtered = df_filtered[df_filtered['Maturity'].isin(maturities)]
        
        # Éditeur de données - Afficher uniquement les colonnes de base
        st.markdown("### Modifier ou Supprimer une position")
        st.info("Editez directement les valeurs dans le tableau ci-dessous. Les modifications seront sauvegardees automatiquement.")
        
        edited_df = st.data_editor(
            df_filtered,
            use_container_width=True,
            height=400,
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "Position_Size": st.column_config.NumberColumn("Position Size", format="%.0f"),
                "Maturity": st.column_config.TextColumn("Maturity"),
                "Settlement_Price": st.column_config.NumberColumn("Settlement Price", format="%.2f"),
                "Contract_Size": st.column_config.NumberColumn("Contract Size", format="%.0f"),
                "Type": st.column_config.SelectboxColumn("Type", options=["C", "P"])
            }
        )
        
        # Sauvegarder les modifications
        if not edited_df.equals(df_filtered):
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Sauvegarder les modifications", use_container_width=True):
                    # Mettre à jour les positions avec les modifications
                    for idx in edited_df.index:
                        if idx in df.index:
                            st.session_state.positions.loc[idx] = edited_df.loc[idx]
                    
                    # Supprimer les lignes supprimées
                    deleted_indices = set(df_filtered.index) - set(edited_df.index)
                    if deleted_indices:
                        st.session_state.positions = st.session_state.positions.drop(deleted_indices).reset_index(drop=True)
                    
                    save_data()
                    st.success("Modifications sauvegardees")
                    st.rerun()
    else:
        st.info("Aucune position enregistree. Ajoutez une position ou importez un fichier Excel.")

# Onglet Greeks
if selected_tab == "Greeks":
    st.subheader("Analyse des Greeks")
    
    # Modèle B76
    st.markdown("### Black-76 Model")
    if not st.session_state.b76_greeks.empty:
        st.dataframe(
            st.session_state.b76_greeks.style.format({
                'Position_Size': '{:.0f}',
                'Bid': '{:.4f}',
                'Ask': '{:.4f}',
                'Last_Price': '{:.4f}',
                'IV': '{:.4f}',
                'Delta': '{:.4f}',
                'Gamma': '{:.4f}',
                'Vega': '{:.4f}',
                'Theta': '{:.4f}',
                'Rho': '{:.4f}',
                'PnL': '{:.2f}'
            }),
            use_container_width=True,
            height=200
        )
    else:
        st.info("Aucune donnee B76. Cliquez sur Actualiser pour calculer.")
    
    st.markdown("---")
    
    # Modèle Bachelier
    st.markdown("### Bachelier Model")
    if not st.session_state.bachelier_greeks.empty:
        st.dataframe(
            st.session_state.bachelier_greeks.style.format({
                'Position_Size': '{:.0f}',
                'Bid': '{:.4f}',
                'Ask': '{:.4f}',
                'Last_Price': '{:.4f}',
                'IV': '{:.4f}',
                'Delta': '{:.4f}',
                'Gamma': '{:.4f}',
                'Vega': '{:.4f}',
                'Theta': '{:.4f}',
                'Rho': '{:.4f}',
                'PnL': '{:.2f}'
            }),
            use_container_width=True,
            height=200
        )
    else:
        st.info("Aucune donnee Bachelier. Cliquez sur Actualiser pour calculer.")
    
    st.markdown("---")
    
    # Modèle Local Volatility
    st.markdown("### Local Volatility Model")
    if not st.session_state.local_vol_greeks.empty:
        st.dataframe(
            st.session_state.local_vol_greeks.style.format({
                'Position_Size': '{:.0f}',
                'Bid': '{:.4f}',
                'Ask': '{:.4f}',
                'Last_Price': '{:.4f}',
                'IV': '{:.4f}',
                'Delta': '{:.4f}',
                'Gamma': '{:.4f}',
                'Vega': '{:.4f}',
                'Theta': '{:.4f}',
                'Rho': '{:.4f}',
                'PnL': '{:.2f}'
            }),
            use_container_width=True,
            height=200
        )
    else:
        st.info("Aucune donnee Local Vol. Cliquez sur Actualiser pour calculer.")
    
    st.markdown("---")
    
    # Modèle Heston
    st.markdown("### Heston Model")
    if not st.session_state.heston_greeks.empty:
        st.dataframe(
            st.session_state.heston_greeks.style.format({
                'Position_Size': '{:.0f}',
                'Bid': '{:.4f}',
                'Ask': '{:.4f}',
                'Last_Price': '{:.4f}',
                'IV': '{:.4f}',
                'Delta': '{:.4f}',
                'Gamma': '{:.4f}',
                'Vega': '{:.4f}',
                'Theta': '{:.4f}',
                'Rho': '{:.4f}',
                'PnL': '{:.2f}'
            }),
            use_container_width=True,
            height=200
        )
    else:
        st.info("Aucune donnee Heston. Cliquez sur Actualiser pour calculer.")

# Onglet Courbe des Taux
if selected_tab == "Courbe des Taux":
    st.subheader("Courbe des Taux US Treasury")
    
    if st.session_state.rates_data is not None:
        rates_df = st.session_state.rates_data
        
        if 'Tenor' in rates_df.columns and 'Rate_%' in rates_df.columns:
            fig_rates = go.Figure()
            
            fig_rates.add_trace(go.Scatter(
                x=rates_df['Tenor'],
                y=rates_df['Rate_%'],
                mode='lines+markers',
                name='Yield Curve',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig_rates.update_layout(
                title='US Treasury Yield Curve',
                xaxis_title='Maturity (Years)',
                yaxis_title='Yield (%)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_rates, use_container_width=True)
            st.dataframe(rates_df, use_container_width=True)
    else:
        st.info("Aucune donnee de courbe des taux. Importez un fichier Excel contenant l'onglet 'US_Rates_Curve'.")

# Onglet Analyse
if selected_tab == "Analyse":
    st.subheader("Analyse de Risque")
    
    if not df.empty:
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            if 'Type' in df.columns and 'Position_Size' in df.columns:
                put_call_dist = df.groupby('Type')['Position_Size'].sum().reset_index()
                fig_pc = px.pie(
                    put_call_dist,
                    values='Position_Size',
                    names='Type',
                    title='Distribution Put/Call',
                    color='Type',
                    color_discrete_map={'C': '#2ecc71', 'P': '#e74c3c'}
                )
                st.plotly_chart(fig_pc, use_container_width=True)
        
        with analysis_col2:
            if 'Strike' in df.columns and 'Position_Size' in df.columns:
                position_by_strike = df.groupby('Strike')['Position_Size'].sum().reset_index()
                fig_strike = px.bar(
                    position_by_strike,
                    x='Strike',
                    y='Position_Size',
                    title='Position par Strike',
                    labels={'Strike': 'Strike Price', 'Position_Size': 'Position'}
                )
                st.plotly_chart(fig_strike, use_container_width=True)
    else:
        st.info("Aucune donnee disponible pour l'analyse.")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center'>
        <p>Risk Monitor v2.0 | Derniere mise a jour : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """,
    unsafe_allow_html=True
)