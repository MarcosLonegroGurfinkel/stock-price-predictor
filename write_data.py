class DataWritter:
    def __init__(self, data, name):
        self.data = data
        self.name = name

    def write_data_to_csv(self, save_to_csv=None, csv_file_name=None):
        """Ask if data should be saved to an excel, if yes then ask the user to input csv name and create the file
        at the project's root directory."""
        if not save_to_csv:
            save_to_csv = input(f"Do you want to save '{self.name}' to a csv file: [y/N]: ")
        if save_to_csv.lower() in ["y", "yes"]:
            if not csv_file_name:
                csv_file_name = input("Enter the csv file name (remember to include the .csv): ")
            if csv_file_name[-4:] != ".csv":
                csv_file_name = csv_file_name.replace(".", "")
                csv_file_name += ".csv"
            self.data.to_csv(csv_file_name, index=False)