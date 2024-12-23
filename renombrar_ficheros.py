import os

class FileRenamer:
    def __init__(self, directory: str, prefix: str):
        """
        Inicializa el objeto FileRenamer.

        :param directory: Ruta del directorio donde se encuentran los archivos.
        :param prefix: Prefijo que se agregará a los archivos.
        """
        self.directory = directory
        self.prefix = prefix

    def rename_files(self):
        """
        Renombra los archivos en el directorio añadiendo el prefijo configurado.
        """
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"El directorio '{self.directory}' no existe.")

        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"La ruta '{self.directory}' no es un directorio válido.")

        files = os.listdir(self.directory)
        for file_name in files:
            file_path = os.path.join(self.directory, file_name)
            if os.path.isfile(file_path):  # Ignora subdirectorios
                new_name = f"{self.prefix}{file_name}"
                new_path = os.path.join(self.directory, new_name)
                os.rename(file_path, new_path)
                print(f"Renombrado: '{file_name}' -> '{new_name}'")


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del directorio y prefijo deseado
    directorio = "recursos_reconocimiento_facial/Data/Cristian-15-12-2024"
    prefijo = "otr_01"

    renamer = FileRenamer(directorio, prefijo)
    renamer.rename_files()
