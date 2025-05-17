import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import CharacterNetworkGenerator ,NamedEntityRecognizer
def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path,save_path)
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )
    return output_chart

def get_character_network(subtitles_path,ner_path): 
    ner = NamedEntityRecognizer()
    df = ner.get_ners(subtitles_path,ner_path)
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(df)
    network_html = character_network_generator.draw_network_graph(relationship_df)
    return network_html

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification with zero shot<h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list_str = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles/Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes,
                                                inputs=[theme_list_str,subtitles_path,save_path],
                                                outputs=[plot])
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network Generator<h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles/Script Path")
                        ner_path = gr.Textbox(label="NERs Save Path")
                        get_network_button = gr.Button("Get Character Network")
                        get_network_button.click(get_character_network,
                                                inputs=[subtitles_path,ner_path],
                                                outputs=[network_html])
    iface.launch(share=True)

if __name__ == "__main__":
    main()  