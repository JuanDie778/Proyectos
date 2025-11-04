# remove_excel_rows.py
import pandas as pd
import os
from datetime import datetime
import shutil

class ExcelRowRemover:
    def __init__(self, excel_file_path: str):
        self.excel_file = excel_file_path
        self.backup_folder = "excel_backups"
        
        # Crear carpeta de backups si no existe
        if not os.path.exists(self.backup_folder):
            os.makedirs(self.backup_folder)
    
    def create_backup(self):
        """Crea un backup del archivo Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{os.path.basename(self.excel_file)}"
        backup_path = os.path.join(self.backup_folder, backup_name)
        
        shutil.copy2(self.excel_file, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        return backup_path
    
    def show_file_info(self):
        """Muestra información del archivo Excel"""
        if not os.path.exists(self.excel_file):
            print(f"❌ El archivo {self.excel_file} no existe")
            return None
        
        try:
            df = pd.read_excel(self.excel_file)
            print(f"\n📊 INFORMACIÓN DEL ARCHIVO:")
            print(f"📍 Archivo: {self.excel_file}")
            print(f"📈 Total de filas: {len(df)}")
            print(f"🏷️ Total de columnas: {len(df.columns)}")
            print(f"📋 Columnas: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error leyendo el archivo: {e}")
            return None
    
    def show_preview(self, num_rows: int = 10):
        """Muestra una preview del archivo"""
        df = self.show_file_info()
        if df is not None:
            print(f"\n👀 PREVIEW (primeras {num_rows} filas):")
            print("=" * 80)
            print(df.head(num_rows))
            print("=" * 80)
    
    def remove_rows(self, rows_to_remove: list, create_backup: bool = True):
        """
        Elimina filas específicas del Excel
        
        Args:
            rows_to_remove: Lista de números de fila a eliminar (comenzando desde 0)
            create_backup: Si True, crea un backup antes de modificar
        """
        if not os.path.exists(self.excel_file):
            print(f"❌ El archivo {self.excel_file} no existe")
            return False
        
        try:
            # Leer el archivo Excel
            df = pd.read_excel(self.excel_file)
            total_rows_before = len(df)
            
            # Validar que las filas a eliminar existan
            valid_rows = [row for row in rows_to_remove if row < total_rows_before]
            invalid_rows = [row for row in rows_to_remove if row >= total_rows_before]
            
            if invalid_rows:
                print(f"⚠️ Advertencia: Las siguientes filas no existen y serán ignoradas: {invalid_rows}")
            
            if not valid_rows:
                print("❌ No hay filas válidas para eliminar")
                return False
            
            # Crear backup si se solicita
            if create_backup:
                self.create_backup()
            
            # Eliminar las filas
            df_cleaned = df.drop(valid_rows, errors='ignore')
            rows_removed = total_rows_before - len(df_cleaned)
            
            # Guardar el nuevo archivo
            df_cleaned.to_excel(self.excel_file, index=False)
            
            print(f"\n✅ ELIMINACIÓN COMPLETADA:")
            print(f"📊 Filas antes: {total_rows_before}")
            print(f"📊 Filas después: {len(df_cleaned)}")
            print(f"🗑️ Filas eliminadas: {rows_removed}")
            print(f"💾 Archivo actualizado: {self.excel_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante la eliminación: {e}")
            return False
    
    def remove_rows_range(self, start_row: int, end_row: int, create_backup: bool = True):
        """Elimina un rango de filas"""
        rows_to_remove = list(range(start_row, end_row + 1))
        return self.remove_rows(rows_to_remove, create_backup)
    
    def remove_first_n_rows(self, n: int, create_backup: bool = True):
        """Elimina las primeras N filas"""
        return self.remove_rows_range(0, n - 1, create_backup)
    
    def remove_last_n_rows(self, n: int, create_backup: bool = True):
        """Elimina las últimas N filas"""
        if not os.path.exists(self.excel_file):
            print(f"❌ El archivo {self.excel_file} no existe")
            return False
        
        try:
            df = pd.read_excel(self.excel_file)
            total_rows = len(df)
            start_row = max(0, total_rows - n)
            return self.remove_rows_range(start_row, total_rows - 1, create_backup)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def main():
    """Función principal interactiva"""
    print("=" * 60)
    print("🗑️  ELIMINADOR DE FILAS EXCEL")
    print("=" * 60)
    
    # Archivo por defecto (puedes cambiarlo)
    default_file = "linkedin_bim_posts_consolidated.xlsx"
    excel_file = input(f"📁 Ruta del archivo Excel [{default_file}]: ").strip()
    
    if not excel_file:
        excel_file = default_file
    
    # Crear instancia del removedor
    remover = ExcelRowRemover(excel_file)
    
    # Mostrar información del archivo
    df = remover.show_file_info()
    if df is None:
        return
    
    # Mostrar preview
    preview = input("\n👀 ¿Ver preview del archivo? (s/n) [s]: ").strip().lower()
    if preview != 'n':
        remover.show_preview(10)
    
    # Opciones de eliminación
    print(f"\n🎯 OPCIONES DE ELIMINACIÓN:")
    print("1. Eliminar filas específicas (ej: 0,5,7,10)")
    print("2. Eliminar rango de filas (ej: 0-19)")
    print("3. Eliminar primeras N filas")
    print("4. Eliminar últimas N filas")
    
    option = input("\n🔢 Elige una opción (1-4): ").strip()
    
    if option == "1":
        # Eliminar filas específicas
        rows_input = input("📝 Ingresa los números de fila a eliminar (separados por comas): ").strip()
        try:
            rows_to_remove = [int(row.strip()) for row in rows_input.split(",")]
            confirm = input(f"⚠️ ¿Eliminar las filas {rows_to_remove}? (s/n): ").strip().lower()
            if confirm == 's':
                remover.remove_rows(rows_to_remove, create_backup=True)
            else:
                print("❌ Operación cancelada")
        except ValueError:
            print("❌ Formato incorrecto. Usa números separados por comas.")
    
    elif option == "2":
        # Eliminar rango de filas
        try:
            range_input = input("📝 Ingresa el rango (ej: 0-19): ").strip()
            start, end = map(int, range_input.split('-'))
            confirm = input(f"⚠️ ¿Eliminar filas desde {start} hasta {end}? (s/n): ").strip().lower()
            if confirm == 's':
                remover.remove_rows_range(start, end, create_backup=True)
            else:
                print("❌ Operación cancelada")
        except ValueError:
            print("❌ Formato incorrecto. Usa formato: inicio-fin")
    
    elif option == "3":
        # Eliminar primeras N filas
        try:
            n = int(input("📝 ¿Cuántas primeras filas quieres eliminar?: ").strip())
            confirm = input(f"⚠️ ¿Eliminar las primeras {n} filas? (s/n): ").strip().lower()
            if confirm == 's':
                remover.remove_first_n_rows(n, create_backup=True)
            else:
                print("❌ Operación cancelada")
        except ValueError:
            print("❌ Debes ingresar un número válido")
    
    elif option == "4":
        # Eliminar últimas N filas
        try:
            n = int(input("📝 ¿Cuántas últimas filas quieres eliminar?: ").strip())
            confirm = input(f"⚠️ ¿Eliminar las últimas {n} filas? (s/n): ").strip().lower()
            if confirm == 's':
                remover.remove_last_n_rows(n, create_backup=True)
            else:
                print("❌ Operación cancelada")
        except ValueError:
            print("❌ Debes ingresar un número válido")
    
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()