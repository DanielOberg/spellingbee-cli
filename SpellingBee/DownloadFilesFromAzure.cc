//
//  DownloadFilesFromAzure.cpp
//  SpellingBee
//
//  Created by Daniel Oberg on 2017-06-27.
//  Copyright © 2017 United Lambdas. All rights reserved.
//

#include "DownloadFilesFromAzure.hh"

#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <boost/filesystem/fstream.hpp>

#include <was/storage_account.h>
#include <was/blob.h>

#include <string>

const utility::string_t storage_connection_string(U("DefaultEndpointsProtocol=https;AccountName=brutaljapanese;AccountKey=7W04u3TDzNie7B8wZboHNOflPhKcRrxfyF8Bx/7D21dSa9ewMQrAtYtSBc1blyMuUuUb8ffTcWy+JPSWlnD7WQ==;EndpointSuffix=core.windows.net"));

void downloadAll() {
    
    std::ofstream f("train_list.csv");
    
    f << "Sample Filename,Phonetics" << std::endl;
    
    for (const auto romanjiAndHiragana : romajiToHiragana ) {
        // Retrieve storage account from connection string.
        azure::storage::cloud_storage_account storage_account = azure::storage::cloud_storage_account::parse(storage_connection_string);
        
        // Create the blob client.
        azure::storage::cloud_blob_client blob_client = storage_account.create_cloud_blob_client();
        
        // Retrieve a reference to a previously created container.
        std::stringstream containerName;
        containerName << "raw-" << romanjiAndHiragana.first;
        azure::storage::cloud_blob_container container = blob_client.get_container_reference(U(containerName.str()));
        
        // Output URI of each item.
        azure::storage::list_blob_item_iterator end_of_results;
        for (auto it = container.list_blobs(); it != end_of_results; ++it)
        {
            if (it->is_blob())
            {
                std::cout << "Blob: " << it->as_blob().uri().primary_uri().to_string() << std::endl;
                // Retrieve reference to a blob named "my-blob-1".
                azure::storage::cloud_block_blob blockBlob = it->as_blob();
                
                std::stringstream path;
                path << "./raw/" << it->as_blob().name() << ".pcm";
                f << it->as_blob().name() << ".pcm," << romanjiAndHiragana.first << std::endl;
                
                if (boost::filesystem::exists(path.str())) continue;
                
                // Save blob contents to a file.
                concurrency::streams::container_buffer<std::vector<int8_t>> buffer;
                concurrency::streams::ostream output_stream(buffer);
                blockBlob.download_to_stream(output_stream);
                
                std::cout << path.str();
                
                std::ofstream outfile(path.str(), std::ofstream::binary);
                auto &data = buffer.collection();
                
                outfile.write((char *)&data[0], buffer.size());
                outfile.close();
            }
            else
            {
//                std::wcout << U("Directory: ") << it->as_directory().uri().primary_uri().to_string() << std::endl;
            }
        }
    }
    f.close();

}
