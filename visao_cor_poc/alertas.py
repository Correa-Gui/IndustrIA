import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List

class GerenciadorAlertas:
    """POC: Alertas via Banner Streamlit + SMTP"""

    def __init__(self, smtp_config: Dict = None):
        self.smtp_config = smtp_config or {
            'host': 'smtp.gmail.com',
            'port': 587,
            'usuario': '',
            'senha': '',
            'de': 'noreply@visao-cor.com',
            'para': []
        }
        self.alertas_buffer = []
        self.nivel_alerta = 'INFO'

    def registrar_alerta(self, nivel: str, mensagem: str, dados: Dict = None):
        """Registra alerta em buffer"""
        alerta = {
            'nivel': nivel,
            'mensagem': mensagem,
            'dados': dados or {},
        }
        self.alertas_buffer.append(alerta)
        self.nivel_alerta = nivel

    def verificar_threshold(self, delta_e: float, threshold: float = 5.0):
        """Verifica se delta_e ultrapassou threshold"""
        if delta_e > threshold:
            self.registrar_alerta(
                'WARNING',
                f'Delta E {delta_e:.2f} acima do threshold {threshold}',
                {'delta_e': delta_e, 'threshold': threshold}
            )
            return True
        return False

    def enviar_email(self, destinatarios: List[str], assunto: str, mensagem: str) -> bool:
        """Envia e-mail via SMTP"""
        if not self.smtp_config['usuario'] or not self.smtp_config['senha']:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['de']
            msg['To'] = ', '.join(destinatarios)
            msg['Subject'] = assunto

            msg.attach(MIMEText(mensagem, 'plain'))

            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['usuario'], self.smtp_config['senha'])
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            self.registrar_alerta('ERROR', f'Falha ao enviar e-mail: {str(e)}')
            return False

    def gerar_banner_streamlit(self) -> str:
        """Gera texto para banner Streamlit"""
        if not self.alertas_buffer:
            return ''

        mensagens = []
        for alerta in self.alertas_buffer:
            icon = {'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️'}.get(alerta['nivel'], '📌')
            mensagens.append(f"{icon} [{alerta['nivel']}] {alerta['mensagem']}")

        return '\n'.join(mensagens)

    def limpar_buffer(self):
        """Limpa buffer de alertas"""
        self.alertas_buffer = []


if __name__ == '__main__':
    alertas = GerenciadorAlertas()
    alertas.registrar_alerta('INFO', 'Sistema iniciado')
    alertas.verificar_threshold(6.5, threshold=5.0)

    banner = alertas.gerar_banner_streamlit()
    print(banner)
